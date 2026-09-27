use crate::debank::{
    BlockFile, BlockStorageDiff, DebankBlock, DebankOutPut, DebankTransaction, build_debank_traces,
    build_genesis_txs_and_traces, get_storage_contracts_from_bundle,
    get_storage_contracts_from_genesis, get_storage_diffs_from_changesets,
};
use alloy_consensus::{BlockHeader, transaction::TxHashRef};
use alloy_eips::{BlockId, eip2718::Encodable2718};
use alloy_evm::evm::EvmFactoryExt;
use alloy_network::ReceiptResponse;
use alloy_primitives::U256;
use alloy_rpc_types_eth::Header;
use async_trait::async_trait;
use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use op_alloy_consensus::OpReceipt;
use op_alloy_rpc_types::OpTransactionReceipt;
use op_revm::{L1BlockInfo, OpSpecId};
use reth_chainspec::{ChainSpecProvider, EthChainSpec};
use reth_evm::ConfigureEvm;
use reth_optimism_evm::{extract_l1_info, revm_spec_by_timestamp_after_bedrock};
use reth_optimism_forks::OpHardforks;
use reth_primitives_traits::{BlockBody, RecoveredBlock};
use reth_revm::{database::StateProviderDatabase, db::State};
use reth_rpc_eth_api::{
    EthApiTypes, FromEthApiError, RpcNodeCore, RpcTypes,
    helpers::{EthBlocks, LoadReceipt, TraceExt},
};
use reth_rpc_eth_types::{EthApiError, cache::db::StateCacheDb};
use reth_storage_api::{ChangeSetReader, StorageChangeSetReader};
use revm::{context_interface::Block, database::states::bundle_state::BundleRetention};
use revm_bytecode::opcode::OpCode;
use revm_inspectors::tracing::{OpcodeFilter, TracingInspector, TracingInspectorConfig};

#[rpc(server, namespace = "trace")]
pub trait OpDebankTraceApi {
    #[method(name = "debankBlock")]
    async fn debank_block(&self, block_id: BlockId) -> RpcResult<DebankOutPut>;
}

#[derive(Debug)]
pub struct OpDebankTraceApiImpl<Eth> {
    eth: Eth,
}

impl<Eth> OpDebankTraceApiImpl<Eth> {
    pub fn new(eth: Eth) -> Self {
        Self { eth }
    }
}

fn get_deposit_nonce(receipt: &OpTransactionReceipt) -> Option<u64> {
    if let OpReceipt::Deposit(dep) = &receipt.inner.inner.receipt {
        dep.deposit_nonce
    } else {
        None
    }
}

fn get_l1_fee(receipt: &OpTransactionReceipt) -> Option<u128> {
    receipt.l1_block_info.l1_fee
}

/// Operator fee charged on top of L2 gas since Isthmus, using the same function as execution.
fn get_operator_fee(
    l1_block_info: &L1BlockInfo,
    encoded_tx: &[u8],
    gas_used: u64,
    spec: OpSpecId,
) -> U256 {
    // The Isthmus activation block still carries Ecotone-format L1 info without operator fee
    // params; the fee is zero there.
    if !spec.is_enabled_in(OpSpecId::ISTHMUS) ||
        l1_block_info.operator_fee_scalar.is_none() ||
        l1_block_info.operator_fee_constant.is_none()
    {
        return U256::ZERO;
    }
    l1_block_info.operator_fee_charge(encoded_tx, U256::from(gas_used), spec)
}

/// Fees the sender paid outside L2 gas: the L1 data fee and the operator fee.
fn get_fee_outside_gas(
    receipt: &OpTransactionReceipt,
    encoded_tx: &[u8],
    l1_block_info: &L1BlockInfo,
    spec: OpSpecId,
) -> U256 {
    U256::from(get_l1_fee(receipt).unwrap_or_default()) +
        get_operator_fee(l1_block_info, encoded_tx, receipt.gas_used(), spec)
}

fn ensure_receipt_count(tx_count: usize, receipt_count: usize) -> Result<(), EthApiError> {
    if tx_count != receipt_count {
        return Err(EthApiError::EvmCustom(format!(
            "transaction/receipt count mismatch: {tx_count} transactions, {receipt_count} receipts"
        )));
    }
    Ok(())
}

fn ensure_changeset_coverage(
    block_number: u64,
    has_account_changes: bool,
    has_storage_changes: bool,
    account_changeset_count: usize,
    storage_changeset_count: usize,
) -> Result<(), EthApiError> {
    if has_account_changes && account_changeset_count == 0 {
        return Err(EthApiError::EvmCustom(format!(
            "account changesets unavailable for block {block_number}"
        )));
    }
    if has_storage_changes && storage_changeset_count == 0 {
        return Err(EthApiError::EvmCustom(format!(
            "storage changesets unavailable for block {block_number}"
        )));
    }
    Ok(())
}

#[async_trait]
impl<Eth> OpDebankTraceApiServer for OpDebankTraceApiImpl<Eth>
where
    Eth: TraceExt + EthBlocks + LoadReceipt + 'static,
    Eth: RpcNodeCore,
    <Eth as EthApiTypes>::NetworkTypes: RpcTypes<Receipt = OpTransactionReceipt>,
    <Eth as RpcNodeCore>::Provider: ChainSpecProvider<ChainSpec: EthChainSpec + OpHardforks>
        + ChangeSetReader
        + StorageChangeSetReader,
{
    async fn debank_block(&self, block_id: BlockId) -> RpcResult<DebankOutPut> {
        Ok(self.trace_debank_block_inner(block_id).await.map_err(Into::into)?)
    }
}

impl<Eth> OpDebankTraceApiImpl<Eth>
where
    Eth: TraceExt + EthBlocks + LoadReceipt + 'static,
    Eth: RpcNodeCore,
    <Eth as EthApiTypes>::NetworkTypes: RpcTypes<Receipt = OpTransactionReceipt>,
    <Eth as RpcNodeCore>::Provider: ChainSpecProvider<ChainSpec: EthChainSpec + OpHardforks>
        + ChangeSetReader
        + StorageChangeSetReader,
{
    async fn trace_debank_block_inner(
        &self,
        block_id: BlockId,
    ) -> Result<DebankOutPut, Eth::Error> {
        let eth = &self.eth;

        let block = eth.recovered_block(block_id).await?;
        let Some(block) = block else {
            return Err(EthApiError::HeaderNotFound(block_id).into());
        };
        let block_id = block.hash().into();

        let debank_block: DebankBlock = block.as_ref().into();
        let debank_header = build_rpc_header(&block);

        if block.number() == 0 {
            let chain_spec = reth_rpc_eth_api::RpcNodeCore::provider(eth).chain_spec();
            let genesis = chain_spec.genesis();
            let mut state_diff: BlockStorageDiff = genesis.into();
            state_diff.hash = block.state_root();
            let (transactions, traces) = build_genesis_txs_and_traces(genesis);
            let block_file = BlockFile {
                block: debank_block,
                transactions,
                traces,
                storage_contracts: get_storage_contracts_from_genesis(genesis),
                ..Default::default()
            };
            let validation_hash = block_file.validation().validation_hash;
            return Ok(DebankOutPut {
                block_file,
                header: debank_header,
                state_diff: alloy_rlp::encode(state_diff).into(),
                validation_hash,
            });
        }

        let receipts: Option<Vec<OpTransactionReceipt>> = eth.block_receipts(block_id).await?;
        let Some(receipts) = receipts else {
            return Err(EthApiError::HeaderNotFound(block_id).into());
        };

        let chain_spec = reth_rpc_eth_api::RpcNodeCore::provider(eth).chain_spec();
        let spec = revm_spec_by_timestamp_after_bedrock(&chain_spec, block.timestamp());
        let l1_block_info = extract_l1_info(block.body()).map_err(|err| {
            EthApiError::EvmCustom(format!("failed to extract L1 block info: {err}"))
        })?;

        let transactions = block.body().transactions();
        ensure_receipt_count(transactions.len(), receipts.len())?;
        let mut debank_txs: Vec<DebankTransaction> = Vec::with_capacity(transactions.len());
        for (tx, receipt) in transactions.iter().zip(&receipts) {
            let deposit_nonce = get_deposit_nonce(receipt);
            let fee_outside_gas =
                get_fee_outside_gas(receipt, &tx.encoded_2718(), &l1_block_info, spec);
            debank_txs.push(DebankTransaction::from((receipt, tx, deposit_nonce, fee_outside_gas)));
        }

        let parent_block = eth.recovered_block(block.parent_hash().into()).await?;
        let Some(parent_block) = parent_block else {
            return Err(EthApiError::HeaderNotFound(block_id).into());
        };

        let mut block_file =
            BlockFile { block: debank_block, transactions: debank_txs, ..Default::default() };

        let (trace_results, mut state_diff, change_addresses) =
            trace_all_block(eth, block_id).await?;

        for (trace, error_trace, event, error_event) in trace_results {
            block_file.traces.extend(trace);
            block_file.error_traces.extend(error_trace);
            block_file.events.extend(event);
            block_file.error_events.extend(error_event);
        }
        state_diff.hash = block.state_root();
        state_diff.parent_hash = parent_block.state_root();
        block_file.storage_contracts = change_addresses;
        let validation_hash = block_file.validation().validation_hash;

        Ok(DebankOutPut {
            block_file,
            header: debank_header,
            state_diff: alloy_rlp::encode(state_diff).into(),
            validation_hash,
        })
    }
}

fn build_rpc_header<B>(block: &RecoveredBlock<B>) -> Header
where
    B: reth_primitives_traits::Block,
{
    Header {
        inner: alloy_consensus::Header {
            parent_hash: block.header().parent_hash(),
            ommers_hash: block.header().ommers_hash(),
            beneficiary: block.header().beneficiary(),
            state_root: block.header().state_root(),
            transactions_root: block.header().transactions_root(),
            receipts_root: block.header().receipts_root(),
            logs_bloom: block.header().logs_bloom(),
            difficulty: block.header().difficulty(),
            number: block.header().number(),
            gas_limit: block.header().gas_limit(),
            gas_used: block.header().gas_used(),
            timestamp: block.header().timestamp(),
            extra_data: block.header().extra_data().clone(),
            mix_hash: block.header().mix_hash().unwrap_or_default(),
            nonce: block.header().nonce().unwrap_or_default(),
            base_fee_per_gas: block.header().base_fee_per_gas(),
            withdrawals_root: block.header().withdrawals_root(),
            blob_gas_used: block.header().blob_gas_used(),
            excess_blob_gas: block.header().excess_blob_gas(),
            parent_beacon_block_root: block.header().parent_beacon_block_root(),
            requests_hash: block.header().requests_hash(),
            block_access_list_hash: block.header().block_access_list_hash(),
            slot_number: None,
        },
        hash: block.hash(),
        total_difficulty: None,
        size: None,
    }
}

type TraceEntry = (
    Vec<crate::debank::DebankTrace>,
    Vec<crate::debank::DebankTrace>,
    Vec<crate::debank::DebankEvent>,
    Vec<crate::debank::DebankEvent>,
);

async fn trace_all_block<Eth>(
    eth: &Eth,
    block_id: BlockId,
) -> Result<(Vec<TraceEntry>, BlockStorageDiff, Vec<alloy_primitives::Address>), Eth::Error>
where
    Eth: TraceExt + RpcNodeCore + 'static,
    Eth::Error: FromEthApiError,
    <Eth as RpcNodeCore>::Provider: ChangeSetReader + StorageChangeSetReader,
    reth_evm::BlockEnvFor<Eth::Evm>: Block,
{
    use reth_rpc_eth_types::cache::db::StateProviderTraitObjWrapper;

    let block = eth.recovered_block(block_id);
    let ((evm_env, _), block) = futures::try_join!(eth.evm_env_at(block_id), block)?;

    let Some(block) = block else {
        return Err(EthApiError::EvmCustom(format!("cannot find block {block_id}")).into());
    };

    let parent_hash = block.parent_hash();

    eth.spawn_blocking_io_fut(move |this| async move {
        let block_hash = block.hash();
        let block_number: u64 = evm_env.block_env.number().saturating_to();
        let base_fee = evm_env.block_env.basefee();

        let post_state = this.state_at_block_id(block_hash.into()).await?;
        let exec_state = this.state_at_block_id(parent_hash.into()).await?;

        let mut db: StateCacheDb = State::builder()
            .with_database(StateProviderDatabase::new(StateProviderTraitObjWrapper(exec_state)))
            .with_bundle_update()
            .build();

        this.apply_pre_execution_changes(&block, &mut db)?;

        let log_index_cell = std::cell::RefCell::new(0usize);
        let mut idx = 0u64;

        let mut trace_cfg = TracingInspectorConfig::default_parity()
            .set_steps(true)
            .set_record_logs(true)
            .set_exclude_precompile_calls(false);
        trace_cfg.record_opcodes_filter = Some(OpcodeFilter::new().enabled(OpCode::SSTORE));

        let results: Vec<TraceEntry> = this
            .evm_config()
            .evm_factory()
            .create_tracer(&mut db, evm_env, TracingInspector::new(trace_cfg))
            .try_trace_many(block.transactions_recovered(), |mut ctx| {
                use alloy_rpc_types_eth::TransactionInfo;
                let tx_info = TransactionInfo {
                    hash: Some(*ctx.tx.tx_hash()),
                    index: Some(idx),
                    block_hash: Some(block_hash),
                    block_number: Some(block_number),
                    base_fee: Some(base_fee),
                    block_timestamp: Some(block.timestamp()),
                };
                idx += 1;
                let traces = build_debank_traces(
                    tx_info.hash.unwrap(),
                    ctx.take_inspector().into_traces(),
                    &log_index_cell,
                );
                Ok::<_, Eth::Error>(traces)
            })
            .commit_last_tx()
            .collect::<Result<_, _>>()?;

        db.merge_transitions(BundleRetention::PlainState);
        let bundle = db.take_bundle();
        let change_addresses = get_storage_contracts_from_bundle(&bundle);
        let has_account_changes = bundle.state.values().any(|account| account.is_info_changed());
        let has_storage_changes = bundle
            .state
            .values()
            .any(|account| account.storage.values().any(|slot| slot.is_changed()));
        let account_changesets = this
            .provider()
            .account_block_changeset(block_number)
            .map_err(Eth::Error::from_eth_err)?;
        let storage_changesets =
            this.provider().storage_changeset(block_number).map_err(Eth::Error::from_eth_err)?;
        ensure_changeset_coverage(
            block_number,
            has_account_changes,
            has_storage_changes,
            account_changesets.len(),
            storage_changesets.len(),
        )
        .map_err(Eth::Error::from_eth_err)?;
        let storage_diff = get_storage_diffs_from_changesets(
            account_changesets,
            storage_changesets,
            StateProviderDatabase::new(StateProviderTraitObjWrapper(post_state)),
        )
        .map_err(|err| {
            Eth::Error::from_eth_err(EthApiError::EvmCustom(format!(
                "failed to build state diff from changesets: {err}"
            )))
        })?;
        Ok((results, storage_diff, change_addresses))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_changeset_coverage, ensure_receipt_count, get_fee_outside_gas, get_operator_fee,
    };
    use crate::debank::calculate_gas_price;
    use alloy_primitives::{Address, B256, U256, hex};
    use op_alloy_rpc_types::OpTransactionReceipt;
    use op_revm::{L1BlockInfo, OpSpecId};
    use reth_optimism_evm::parse_l1_info;

    // Only deposit (0x7e) and empty inputs are exempt from the operator fee, so an EIP-1559
    // envelope type byte is enough to stand for a regular transaction.
    const REGULAR_TX: &[u8] = &[0x02];

    // Callers pass a `cumulative_gas_used` different from `gas_used`, so charging the operator
    // fee on the wrong gas field fails the assertions.
    fn receipt(gas_used: u64, cumulative_gas_used: u64, l1_fee: u128) -> OpTransactionReceipt {
        serde_json::from_value(serde_json::json!({
            "type": "0x2",
            "status": "0x1",
            "gasUsed": format!("{gas_used:#x}"),
            "cumulativeGasUsed": format!("{cumulative_gas_used:#x}"),
            "effectiveGasPrice": "0x1",
            "l1Fee": format!("{l1_fee:#x}"),
            "logs": [],
            "logsBloom": format!("0x{}", "0".repeat(512)),
            "transactionHash": B256::ZERO,
            "transactionIndex": "0x1",
            "blockHash": B256::ZERO,
            "blockNumber": "0x1",
            "from": Address::ZERO,
            "to": Address::ZERO,
            "contractAddress": null,
        }))
        .unwrap()
    }

    #[test]
    fn operator_fee_is_folded_into_gas_price() {
        // BOB mainnet block 38,821,851 L1 info input (Jovian, scalar 0, constant 1.5e13), and
        // receipt values of tx 0xc6e0e46057ae5c1d5c251456bdd82ac9dbfe274b22d2eb9b06df8986e551cddf.
        // The OperatorFeeVault balance rose by 15_000_000_000_000 wei in this block.
        let l1_block_info = parse_l1_info(&hex!(
            "3db6be2b001e8480000955060000000000000004000000006ab8f29700000000018dc5200000000000000000000000000000000000000000000000000000000003ea758700000000000000000000000000000000000000000000000000000000003b927cc8d0ea7508cfe24b335f09e985abf1e80cc7bc6996339304fe6698e418be29aa00000000000000000000000008f9f14ff43e112b18c96f0986f28cb1878f1d110000000000000da475abf0000190"
        ))
        .unwrap();
        let receipt = receipt(123_359, 169_661, 1_123_409_073_565);
        let fee = get_fee_outside_gas(&receipt, REGULAR_TX, &l1_block_info, OpSpecId::JOVIAN);
        assert_eq!(fee, U256::from(1_123_409_073_565u64 + 15_000_000_000_000));
        assert_eq!(calculate_gas_price(57_611_263, 123_359, fee), U256::from(188_314_406));

        // The pipeline published 66_718_090 for this tx when only the L1 fee was folded in.
        let l1_fee = U256::from(1_123_409_073_565u64);
        assert_eq!(calculate_gas_price(57_611_263, 123_359, l1_fee), U256::from(66_718_090));
    }

    #[test]
    fn zero_operator_fee_params_keep_gas_price() {
        // RISE mainnet block 20,663,202 L1 info input (operator fee params are zero), and receipt
        // values of tx 0xca26bd90e54a8a9d3054979d51f3e2fdd6bf7316e81a7e83d5724930d4e16ce9.
        let l1_block_info = parse_l1_info(&hex!(
            "3db6be2b00000000000000000000000000000003000000006a96d12f00000000018af044000000000000000000000000000000000000000000000000000000000e3d38200000000000000000000000000000000000000000000000000000000000bc6dc7e900ef1ecebbd25413b179d043ef0bae37023eae716e748837f92929a4c5ee380000000000000000000000000ae4b35f7f5efeb4c651684e1bca12993dcbb67d0000000000000000000000000000"
        ))
        .unwrap();
        let receipt = receipt(641_659, 687_793, 0);
        let fee = get_fee_outside_gas(&receipt, REGULAR_TX, &l1_block_info, OpSpecId::JOVIAN);
        assert_eq!(fee, U256::ZERO);
        assert_eq!(calculate_gas_price(1_001_001, 641_659, fee), U256::from(1_001_001));
    }

    #[test]
    fn operator_fee_is_charged_on_receipt_gas_used() {
        // A non-zero scalar makes the fee depend on gas: 5 + 21_000 * 3 * 100 + 7.
        let l1_block_info = L1BlockInfo {
            operator_fee_scalar: Some(U256::from(3)),
            operator_fee_constant: Some(U256::from(7)),
            ..Default::default()
        };
        let receipt = receipt(21_000, 60_000, 5);
        let fee = get_fee_outside_gas(&receipt, REGULAR_TX, &l1_block_info, OpSpecId::JOVIAN);
        assert_eq!(fee, U256::from(6_300_012));
    }

    #[test]
    fn operator_fee_follows_execution_rules() {
        let l1_block_info = L1BlockInfo {
            operator_fee_scalar: Some(U256::from(2)),
            operator_fee_constant: Some(U256::from(50)),
            ..Default::default()
        };
        // Jovian: gas * scalar * 100 + constant.
        assert_eq!(
            get_operator_fee(&l1_block_info, REGULAR_TX, 10, OpSpecId::JOVIAN),
            U256::from(2_050)
        );
        // Isthmus: gas * scalar / 1e6 + constant.
        assert_eq!(
            get_operator_fee(&l1_block_info, REGULAR_TX, 1_000_000, OpSpecId::ISTHMUS),
            U256::from(52)
        );
        assert_eq!(get_operator_fee(&l1_block_info, &[0x7e], 10, OpSpecId::JOVIAN), U256::ZERO);
        assert_eq!(
            get_operator_fee(&l1_block_info, REGULAR_TX, 10, OpSpecId::HOLOCENE),
            U256::ZERO
        );
        // Isthmus activation block: Ecotone-format L1 info carries no operator fee params.
        assert_eq!(
            get_operator_fee(&L1BlockInfo::default(), REGULAR_TX, 10, OpSpecId::ISTHMUS),
            U256::ZERO
        );
        // Jovian activation block: Isthmus-format L1 info, charged with the Jovian formula.
        let mut isthmus_input = [0u8; 176];
        isthmus_input[..4].copy_from_slice(&hex!("098999be"));
        isthmus_input[164..168].copy_from_slice(&2u32.to_be_bytes());
        isthmus_input[168..].copy_from_slice(&50u64.to_be_bytes());
        let l1_block_info = parse_l1_info(&isthmus_input).unwrap();
        assert_eq!(
            get_operator_fee(&l1_block_info, REGULAR_TX, 10, OpSpecId::JOVIAN),
            U256::from(2_050)
        );
    }

    #[test]
    fn rejects_transaction_receipt_count_mismatch() {
        assert!(ensure_receipt_count(2, 1).is_err());
        assert!(ensure_receipt_count(2, 2).is_ok());
    }

    #[test]
    fn rejects_missing_required_changesets() {
        assert!(ensure_changeset_coverage(42, true, false, 0, 1).is_err());
        assert!(ensure_changeset_coverage(42, false, true, 1, 0).is_err());
        assert!(ensure_changeset_coverage(42, true, true, 1, 1).is_ok());
        assert!(ensure_changeset_coverage(42, true, false, 1, 0).is_ok());
        assert!(ensure_changeset_coverage(42, false, true, 0, 1).is_ok());
        assert!(ensure_changeset_coverage(42, false, false, 0, 0).is_ok());
    }
}
