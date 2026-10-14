//! Records call traces for `trace_debankBlock` while blocks are imported, so it can skip
//! re-executing a block this node has just executed.
//!
//! [`DebankEvmConfig`] wraps [`OpEvmConfig`]. EVMs created without an external inspector carry an
//! idle [`DebankCapture`]; the block executor switches it on for each transaction and stores the
//! traces of committed transactions in a [`DebankTraceCache`] when the block finishes.

use alloy_consensus::{BlockHeader, Header, transaction::TxHashRef};
use alloy_evm::{
    Database, Evm, EvmEnv, EvmFactory, RecoveredTx,
    block::{
        BlockExecutionError, BlockExecutionResult, BlockExecutor, BlockExecutorFactory,
        ExecutableTx, GasOutput, OnStateHook, StateDB,
    },
    precompiles::PrecompilesMap,
};
use alloy_op_evm::{
    OpBlockExecutionCtx, OpBlockExecutor, OpEvm, OpEvmContext, OpEvmFactory, OpTx, OpTxError,
    PostExecMode,
    block::OpTxResult,
    post_exec::{PostExecEvm, PostExecExecutedTx, PostExecExecutorExt, PostExecTxContext},
};
use alloy_primitives::{Address, B256, Bytes, TxHash, keccak256};
use op_alloy_consensus::OpTxType;
use op_alloy_rpc_types_engine::OpExecutionData;
use op_revm::{OpHaltReason, OpSpecId};
use reth_chainspec::EthChainSpec;
use reth_evm::{
    ConfigureEngineEvm, ConfigureEvm, EvmEnvFor, ExecutableTxIterator, ExecutionCtxFor,
    execute::BlockBuilder,
};
use reth_optimism_evm::{
    ConfigurePostExecEvm, OpBlockAssembler, OpEvmConfig, OpNextBlockEnvAttributes,
    OpRethReceiptBuilder,
};
use reth_optimism_forks::OpHardforks;
use reth_optimism_payload_builder::OpExecData;
use reth_optimism_primitives::{OpPrimitives, OpReceipt, OpTransactionSigned};
use reth_primitives_traits::{BlockTy, HeaderTy, NodePrimitives, SealedBlock, SealedHeader};
use revm::{
    Inspector,
    context::{
        BlockEnv, CfgEnv,
        result::{EVMError, ResultAndState},
    },
    context_interface::Block,
    database::State,
    inspector::NoOpInspector,
    interpreter::{CallInputs, CallOutcome, CreateInputs, CreateOutcome, Interpreter},
    primitives::Log,
};
use revm_bytecode::opcode::OpCode;
use revm_inspectors::tracing::{
    CallTraceArena, OpcodeFilter, TracingInspector, TracingInspectorConfig,
};
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    sync::{Arc, Mutex},
};

/// Call traces of a block's transactions, in execution order.
pub type BlockTraces = Vec<(TxHash, CallTraceArena)>;

/// Tracer configuration behind `trace_debankBlock`.
pub const fn debank_tracing_config() -> TracingInspectorConfig {
    let mut config = TracingInspectorConfig::default_parity()
        .set_steps(true)
        .set_record_logs(true)
        .set_exclude_precompile_calls(false);
    config.record_opcodes_filter = Some(OpcodeFilter::new().enabled(OpCode::SSTORE));
    config
}

/// Cache key of a block's traces.
///
/// The block hash is only known after execution, so the key covers what decides how the block's
/// transactions execute instead.
pub fn trace_cache_key(
    header: &impl BlockHeader,
    tx_hashes: impl IntoIterator<Item = TxHash>,
) -> B256 {
    let mut buf = Vec::with_capacity(196);
    buf.extend_from_slice(header.parent_hash().as_slice());
    buf.extend_from_slice(header.parent_beacon_block_root().unwrap_or_default().as_slice());
    buf.extend_from_slice(&header.number().to_be_bytes());
    buf.extend_from_slice(&header.timestamp().to_be_bytes());
    buf.extend_from_slice(header.beneficiary().as_slice());
    buf.extend_from_slice(header.mix_hash().unwrap_or_default().as_slice());
    buf.extend_from_slice(&header.base_fee_per_gas().unwrap_or_default().to_be_bytes());
    buf.extend_from_slice(&header.gas_limit().to_be_bytes());
    for hash in tx_hashes {
        buf.extend_from_slice(hash.as_slice());
    }
    keccak256(buf)
}

/// Traces of recently imported blocks, oldest evicted first.
#[derive(Debug)]
pub struct DebankTraceCache {
    capacity: usize,
    entries: Mutex<CacheEntries>,
}

#[derive(Debug, Default)]
struct CacheEntries {
    traces: HashMap<B256, BlockTraces>,
    order: VecDeque<B256>,
}

impl DebankTraceCache {
    /// Creates a cache that holds at most `capacity` blocks.
    pub fn new(capacity: usize) -> Self {
        Self { capacity, entries: Mutex::default() }
    }

    /// Stores the traces of a block, replacing an earlier execution of the same block.
    pub fn insert(&self, key: B256, traces: BlockTraces) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.traces.insert(key, traces).is_none() {
            entries.order.push_back(key);
        }
        while entries.traces.len() > self.capacity {
            let Some(oldest) = entries.order.pop_front() else { break };
            entries.traces.remove(&oldest);
        }
    }

    /// Removes and returns the traces of a block.
    pub fn take(&self, key: &B256) -> Option<BlockTraces> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let traces = entries.traces.remove(key)?;
        entries.order.retain(|k| k != key);
        Some(traces)
    }
}

/// Second half of the inspector of a [`DebankEvm`]: a tracer that runs only while the block
/// executor records a transaction.
#[derive(Debug, Default)]
pub struct DebankCapture {
    /// False when the EVM was created with an external inspector, e.g. by an RPC tracer.
    available: bool,
    tracer: Option<TracingInspector>,
}

impl DebankCapture {
    fn start(&mut self) {
        self.tracer = Some(TracingInspector::new(debank_tracing_config()));
    }

    fn finish(&mut self) -> Option<CallTraceArena> {
        self.tracer.take().map(TracingInspector::into_traces)
    }
}

impl<CTX> Inspector<CTX> for DebankCapture
where
    TracingInspector: Inspector<CTX>,
{
    fn initialize_interp(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if let Some(tracer) = &mut self.tracer {
            tracer.initialize_interp(interp, context);
        }
    }

    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if let Some(tracer) = &mut self.tracer {
            tracer.step(interp, context);
        }
    }

    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if let Some(tracer) = &mut self.tracer {
            tracer.step_end(interp, context);
        }
    }

    fn log(&mut self, context: &mut CTX, log: Log) {
        if let Some(tracer) = &mut self.tracer {
            tracer.log(context, log);
        }
    }

    fn log_full(&mut self, interp: &mut Interpreter, context: &mut CTX, log: Log) {
        if let Some(tracer) = &mut self.tracer {
            tracer.log_full(interp, context, log);
        }
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        self.tracer.as_mut().and_then(|tracer| tracer.call(context, inputs))
    }

    fn call_end(&mut self, context: &mut CTX, inputs: &CallInputs, outcome: &mut CallOutcome) {
        if let Some(tracer) = &mut self.tracer {
            tracer.call_end(context, inputs, outcome);
        }
    }

    fn create(&mut self, context: &mut CTX, inputs: &mut CreateInputs) -> Option<CreateOutcome> {
        self.tracer.as_mut().and_then(|tracer| tracer.create(context, inputs))
    }

    fn create_end(
        &mut self,
        context: &mut CTX,
        inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        if let Some(tracer) = &mut self.tracer {
            tracer.create_end(context, inputs, outcome);
        }
    }

    fn selfdestruct(&mut self, contract: Address, target: Address, value: alloy_primitives::U256) {
        if let Some(tracer) = &mut self.tracer {
            tracer.selfdestruct(contract, target, value);
        }
    }
}

/// [`OpEvm`] whose inspector is paired with a [`DebankCapture`].
#[allow(missing_debug_implementations)]
pub struct DebankEvm<DB: Database, I> {
    inner: OpEvm<DB, (I, DebankCapture), PrecompilesMap, OpTx>,
}

impl<DB, I> DebankEvm<DB, I>
where
    DB: Database,
    I: Inspector<OpEvmContext<DB>>,
{
    fn capture(&mut self) -> &mut DebankCapture {
        &mut self.inner.inspector_mut().1
    }
}

impl<DB, I> Evm for DebankEvm<DB, I>
where
    DB: Database,
    I: Inspector<OpEvmContext<DB>>,
{
    type DB = DB;
    type Tx = OpTx;
    type Error = EVMError<DB::Error, OpTxError>;
    type HaltReason = OpHaltReason;
    type Spec = OpSpecId;
    type BlockEnv = BlockEnv;
    type Precompiles = PrecompilesMap;
    type Inspector = I;

    fn block(&self) -> &BlockEnv {
        self.inner.block()
    }

    fn cfg_env(&self) -> &CfgEnv<OpSpecId> {
        self.inner.cfg_env()
    }

    fn chain_id(&self) -> u64 {
        self.inner.chain_id()
    }

    fn transact_raw(
        &mut self,
        tx: Self::Tx,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        self.inner.transact_raw(tx)
    }

    fn transact_system_call(
        &mut self,
        caller: Address,
        contract: Address,
        data: Bytes,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        self.inner.transact_system_call(caller, contract, data)
    }

    fn finish(self) -> (Self::DB, EvmEnv<Self::Spec, Self::BlockEnv>) {
        self.inner.finish()
    }

    fn set_inspector_enabled(&mut self, enabled: bool) {
        self.inner.set_inspector_enabled(enabled);
    }

    fn components(&self) -> (&Self::DB, &Self::Inspector, &Self::Precompiles) {
        let (db, (inspector, _), precompiles) = self.inner.components();
        (db, inspector, precompiles)
    }

    fn components_mut(&mut self) -> (&mut Self::DB, &mut Self::Inspector, &mut Self::Precompiles) {
        let (db, (inspector, _), precompiles) = self.inner.components_mut();
        (db, inspector, precompiles)
    }
}

impl<DB, I> PostExecEvm for DebankEvm<DB, I>
where
    DB: Database,
    I: Inspector<OpEvmContext<DB>>,
{
    fn begin_post_exec_tx(&mut self, ctx: PostExecTxContext) {
        PostExecEvm::begin_post_exec_tx(&mut self.inner, ctx);
    }

    fn take_last_post_exec_tx_result(&mut self) -> PostExecExecutedTx {
        PostExecEvm::take_last_post_exec_tx_result(&mut self.inner)
    }
}

/// Factory of [`DebankEvm`]s.
#[derive(Debug, Clone, Copy, Default)]
pub struct DebankEvmFactory(OpEvmFactory<OpTx>);

impl EvmFactory for DebankEvmFactory {
    type Evm<DB: Database, I: Inspector<OpEvmContext<DB>>> = DebankEvm<DB, I>;
    type Context<DB: Database> = OpEvmContext<DB>;
    type Tx = OpTx;
    type Error<DBError: core::error::Error + Send + Sync + 'static> = EVMError<DBError, OpTxError>;
    type HaltReason = OpHaltReason;
    type Spec = OpSpecId;
    type BlockEnv = BlockEnv;
    type Precompiles = PrecompilesMap;

    fn create_evm<DB: Database>(
        &self,
        db: DB,
        input: EvmEnv<OpSpecId, BlockEnv>,
    ) -> Self::Evm<DB, NoOpInspector> {
        let capture = DebankCapture { available: true, tracer: None };
        let mut inner = self.0.create_evm_with_inspector(db, input, (NoOpInspector, capture));
        inner.disable_inspector();
        DebankEvm { inner }
    }

    fn create_evm_with_inspector<DB: Database, I: Inspector<Self::Context<DB>>>(
        &self,
        db: DB,
        input: EvmEnv<OpSpecId, BlockEnv>,
        inspector: I,
    ) -> Self::Evm<DB, I> {
        let inner =
            self.0.create_evm_with_inspector(db, input, (inspector, DebankCapture::default()));
        DebankEvm { inner }
    }
}

/// Recording state of one block.
#[derive(Debug)]
struct BlockCapture {
    cache: Arc<DebankTraceCache>,
    parent_hash: B256,
    parent_beacon_block_root: Option<B256>,
    /// Traces of the last executed transaction, until it is committed.
    pending: Option<(TxHash, CallTraceArena)>,
    traces: BlockTraces,
    /// False once a transaction was committed without its traces; the block is then not cached.
    complete: bool,
}

impl BlockCapture {
    fn key(&self, block: &impl Block) -> B256 {
        let header = Header {
            parent_hash: self.parent_hash,
            parent_beacon_block_root: self.parent_beacon_block_root,
            number: block.number().saturating_to(),
            timestamp: block.timestamp().saturating_to(),
            beneficiary: block.beneficiary(),
            mix_hash: block.prevrandao().unwrap_or_default(),
            base_fee_per_gas: Some(block.basefee()),
            gas_limit: block.gas_limit(),
            ..Default::default()
        };
        trace_cache_key(&header, self.traces.iter().map(|(hash, _)| *hash))
    }
}

/// Block executor that keeps the traces of committed transactions for the [`DebankTraceCache`].
#[derive(Debug)]
pub struct DebankBlockExecutor<E> {
    inner: E,
    capture: Option<BlockCapture>,
}

impl<E, DB, I> BlockExecutor for DebankBlockExecutor<E>
where
    E: BlockExecutor<Evm = DebankEvm<DB, I>, Transaction = OpTransactionSigned>,
    DB: Database,
    I: Inspector<OpEvmContext<DB>>,
{
    type Transaction = E::Transaction;
    type Receipt = E::Receipt;
    type Evm = E::Evm;
    type Result = E::Result;

    fn apply_pre_execution_changes(&mut self) -> Result<(), BlockExecutionError> {
        self.inner.apply_pre_execution_changes()
    }

    fn execute_transaction_without_commit(
        &mut self,
        tx: impl ExecutableTx<Self>,
    ) -> Result<Self::Result, BlockExecutionError> {
        let (tx_env, tx) = tx.into_parts();
        let Self { inner, capture } = self;
        let Some(capture) = capture else {
            return inner.execute_transaction_without_commit((tx_env, tx));
        };

        let tx_hash = *TxHashRef::tx_hash(tx.tx());
        let evm = inner.evm_mut();
        evm.capture().start();
        evm.enable_inspector();
        let result = inner.execute_transaction_without_commit((tx_env, tx));
        let evm = inner.evm_mut();
        evm.disable_inspector();
        capture.pending = evm.capture().finish().map(|traces| (tx_hash, traces));
        result
    }

    fn commit_transaction(&mut self, output: Self::Result) -> GasOutput {
        if let Some(capture) = &mut self.capture {
            match capture.pending.take() {
                Some(traces) => capture.traces.push(traces),
                None => capture.complete = false,
            }
        }
        self.inner.commit_transaction(output)
    }

    fn finish(
        self,
    ) -> Result<(Self::Evm, BlockExecutionResult<Self::Receipt>), BlockExecutionError> {
        let Self { inner, capture } = self;
        let capture = capture.filter(|capture| capture.complete);
        let key = capture.as_ref().map(|capture| capture.key(inner.evm().block()));
        let output = inner.finish()?;
        if let (Some(capture), Some(key)) = (capture, key) {
            capture.cache.insert(key, capture.traces);
        }
        Ok(output)
    }

    fn set_state_hook(&mut self, hook: Option<Box<dyn OnStateHook>>) {
        self.inner.set_state_hook(hook);
    }

    fn evm_mut(&mut self) -> &mut Self::Evm {
        self.inner.evm_mut()
    }

    fn evm(&self) -> &Self::Evm {
        self.inner.evm()
    }

    fn receipts(&self) -> &[Self::Receipt] {
        self.inner.receipts()
    }
}

impl<E: PostExecExecutorExt> PostExecExecutorExt for DebankBlockExecutor<E> {
    fn take_post_exec_entries(&mut self) -> Vec<op_alloy_consensus::post_exec::SDMGasEntry> {
        self.inner.take_post_exec_entries()
    }
}

/// [`OpBlockExecutorFactory`](alloy_op_evm::OpBlockExecutorFactory) counterpart that builds
/// [`DebankBlockExecutor`]s.
#[derive(Debug)]
pub struct DebankBlockExecutorFactory<ChainSpec> {
    spec: Arc<ChainSpec>,
    receipt_builder: OpRethReceiptBuilder,
    evm_factory: DebankEvmFactory,
    cache: Option<Arc<DebankTraceCache>>,
}

impl<ChainSpec> Clone for DebankBlockExecutorFactory<ChainSpec> {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec.clone(),
            receipt_builder: self.receipt_builder,
            evm_factory: self.evm_factory,
            cache: self.cache.clone(),
        }
    }
}

impl<ChainSpec> BlockExecutorFactory for DebankBlockExecutorFactory<ChainSpec>
where
    ChainSpec: OpHardforks + 'static,
{
    type EvmFactory = DebankEvmFactory;
    type ExecutionCtx<'a> = OpBlockExecutionCtx;
    type Transaction = OpTransactionSigned;
    type Receipt = OpReceipt;
    type TxExecutionResult = OpTxResult<OpHaltReason, OpTxType>;
    type Executor<'a, DB: StateDB, I: Inspector<<Self::EvmFactory as EvmFactory>::Context<DB>>> =
        DebankBlockExecutor<
            OpBlockExecutor<DebankEvm<DB, I>, &'a OpRethReceiptBuilder, &'a Arc<ChainSpec>>,
        >;

    fn evm_factory(&self) -> &Self::EvmFactory {
        &self.evm_factory
    }

    fn create_executor<'a, DB, I>(
        &'a self,
        mut evm: <Self::EvmFactory as EvmFactory>::Evm<DB, I>,
        ctx: Self::ExecutionCtx<'a>,
    ) -> Self::Executor<'a, DB, I>
    where
        DB: StateDB,
        I: Inspector<<Self::EvmFactory as EvmFactory>::Context<DB>>,
    {
        let capture = match &self.cache {
            Some(cache) if evm.capture().available => Some(BlockCapture {
                cache: cache.clone(),
                parent_hash: ctx.parent_hash,
                parent_beacon_block_root: ctx.parent_beacon_block_root,
                pending: None,
                traces: Vec::new(),
                complete: true,
            }),
            _ => None,
        };
        let inner = OpBlockExecutor::new(evm, ctx, &self.spec, &self.receipt_builder);
        DebankBlockExecutor { inner, capture }
    }
}

/// [`OpEvmConfig`] whose executors record traces into a [`DebankTraceCache`].
#[derive(Debug)]
pub struct DebankEvmConfig<ChainSpec> {
    inner: OpEvmConfig<ChainSpec>,
    executor_factory: DebankBlockExecutorFactory<ChainSpec>,
}

impl<ChainSpec> Clone for DebankEvmConfig<ChainSpec> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), executor_factory: self.executor_factory.clone() }
    }
}

impl<ChainSpec: OpHardforks> DebankEvmConfig<ChainSpec> {
    /// Wraps `inner`; without a cache nothing is recorded.
    pub fn new(inner: OpEvmConfig<ChainSpec>, cache: Option<Arc<DebankTraceCache>>) -> Self {
        let executor_factory = DebankBlockExecutorFactory {
            spec: inner.chain_spec().clone(),
            receipt_builder: *inner.executor_factory.receipt_builder(),
            evm_factory: DebankEvmFactory::default(),
            cache,
        };
        Self { inner, executor_factory }
    }

    /// The cache executors record into.
    pub const fn trace_cache(&self) -> Option<&Arc<DebankTraceCache>> {
        self.executor_factory.cache.as_ref()
    }

    /// Returns the wrapped [`OpEvmConfig`].
    pub fn into_inner(self) -> OpEvmConfig<ChainSpec> {
        self.inner
    }
}

impl<ChainSpec> ConfigureEvm for DebankEvmConfig<ChainSpec>
where
    ChainSpec: EthChainSpec<Header = Header> + OpHardforks + Debug + Send + Sync + Unpin + 'static,
{
    type Primitives = OpPrimitives;
    type Error = <OpEvmConfig<ChainSpec> as ConfigureEvm>::Error;
    type NextBlockEnvCtx = OpNextBlockEnvAttributes;
    type BlockExecutorFactory = DebankBlockExecutorFactory<ChainSpec>;
    type BlockAssembler = OpBlockAssembler<ChainSpec>;

    fn block_executor_factory(&self) -> &Self::BlockExecutorFactory {
        &self.executor_factory
    }

    fn block_assembler(&self) -> &Self::BlockAssembler {
        self.inner.block_assembler()
    }

    fn evm_env(&self, header: &HeaderTy<Self::Primitives>) -> Result<EvmEnvFor<Self>, Self::Error> {
        self.inner.evm_env(header)
    }

    fn next_evm_env(
        &self,
        parent: &HeaderTy<Self::Primitives>,
        attributes: &Self::NextBlockEnvCtx,
    ) -> Result<EvmEnvFor<Self>, Self::Error> {
        self.inner.next_evm_env(parent, attributes)
    }

    fn context_for_block<'a>(
        &self,
        block: &'a SealedBlock<BlockTy<Self::Primitives>>,
    ) -> Result<ExecutionCtxFor<'a, Self>, Self::Error> {
        self.inner.context_for_block(block)
    }

    fn context_for_next_block(
        &self,
        parent: &SealedHeader<HeaderTy<Self::Primitives>>,
        attributes: Self::NextBlockEnvCtx,
    ) -> Result<ExecutionCtxFor<'_, Self>, Self::Error> {
        self.inner.context_for_next_block(parent, attributes)
    }
}

impl<ChainSpec> ConfigureEngineEvm<OpExecutionData> for DebankEvmConfig<ChainSpec>
where
    ChainSpec: EthChainSpec<Header = Header> + OpHardforks + Debug + Send + Sync + Unpin + 'static,
{
    fn evm_env_for_payload(
        &self,
        payload: &OpExecutionData,
    ) -> Result<EvmEnvFor<Self>, Self::Error> {
        self.inner.evm_env_for_payload(payload)
    }

    fn context_for_payload<'a>(
        &self,
        payload: &'a OpExecutionData,
    ) -> Result<ExecutionCtxFor<'a, Self>, Self::Error> {
        self.inner.context_for_payload(payload)
    }

    fn tx_iterator_for_payload(
        &self,
        payload: &OpExecutionData,
    ) -> Result<impl ExecutableTxIterator<Self>, Self::Error> {
        self.inner.tx_iterator_for_payload(payload)
    }
}

impl<ChainSpec> ConfigureEngineEvm<OpExecData> for DebankEvmConfig<ChainSpec>
where
    ChainSpec:
        EthChainSpec<Header = Header> + OpHardforks + Clone + Debug + Send + Sync + Unpin + 'static,
{
    fn evm_env_for_payload(&self, payload: &OpExecData) -> Result<EvmEnvFor<Self>, Self::Error> {
        self.inner.evm_env_for_payload(&payload.0)
    }

    fn context_for_payload<'a>(
        &self,
        payload: &'a OpExecData,
    ) -> Result<ExecutionCtxFor<'a, Self>, Self::Error> {
        self.inner.context_for_payload(&payload.0)
    }

    fn tx_iterator_for_payload(
        &self,
        payload: &OpExecData,
    ) -> Result<impl ExecutableTxIterator<Self>, Self::Error> {
        self.inner.tx_iterator_for_payload(&payload.0)
    }
}

impl<ChainSpec> ConfigurePostExecEvm for DebankEvmConfig<ChainSpec>
where
    ChainSpec: EthChainSpec<Header = Header> + OpHardforks + Debug + Send + Sync + Unpin + 'static,
{
    fn post_exec_executor_for_block<'a, DB: Database>(
        &'a self,
        db: &'a mut State<DB>,
        block: &'a SealedBlock<<Self::Primitives as NodePrimitives>::Block>,
        post_exec_mode: PostExecMode,
    ) -> Result<
        impl BlockExecutor<
            Transaction = <Self::Primitives as NodePrimitives>::SignedTx,
            Receipt = <Self::Primitives as NodePrimitives>::Receipt,
        > + PostExecExecutorExt
        + 'a,
        Self::Error,
    > {
        let evm = self.evm_for_block(db, block.header())?;
        let ctx = self.inner.context_for_block_with_post_exec_mode(block, Some(post_exec_mode));
        Ok(self.create_executor(evm, ctx))
    }

    fn post_exec_builder_for_next_block<'a, DB: Database + 'a>(
        &'a self,
        db: &'a mut State<DB>,
        parent: &'a SealedHeader<<Self::Primitives as NodePrimitives>::BlockHeader>,
        attributes: Self::NextBlockEnvCtx,
        post_exec_mode: PostExecMode,
    ) -> Result<
        impl BlockBuilder<Primitives = Self::Primitives, Executor: PostExecExecutorExt> + 'a,
        Self::Error,
    > {
        let evm_env = self.next_evm_env(parent, &attributes)?;
        let evm = self.evm_with_env(db, evm_env);
        let ctx = self.inner.context_for_next_block_with_post_exec_mode(
            parent,
            attributes,
            post_exec_mode,
        );
        Ok(self.create_block_builder(evm, parent, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debank::build_debank_traces;
    use alloy_consensus::BlockBody;
    use alloy_evm::evm::EvmFactoryExt;
    use alloy_primitives::TxKind;
    use op_alloy_consensus::TxDeposit;
    use reth_optimism_chainspec::{OpChainSpec, OpChainSpecBuilder};
    use reth_primitives_traits::RecoveredBlock;
    use revm::{
        database::{CacheDB, EmptyDB},
        state::{AccountInfo, Bytecode},
    };
    use std::cell::RefCell;

    const CALLER: Address = Address::repeat_byte(0x10);
    const STORE_LOG_CALL: Address = Address::repeat_byte(0xa1);
    const STORE: Address = Address::repeat_byte(0xb2);
    const STORE_REVERT: Address = Address::repeat_byte(0xc3);

    fn contract(code: Vec<u8>) -> AccountInfo {
        let code = Bytecode::new_raw(code.into());
        AccountInfo {
            nonce: 1,
            code_hash: code.hash_slow(),
            code: Some(code),
            ..Default::default()
        }
    }

    fn pre_state() -> CacheDB<EmptyDB> {
        let mut db = CacheDB::new(EmptyDB::default());
        // SSTORE(0, 1); LOG0(0, 0); CALL(0xffff, STORE, 0, 0, 0, 0, 0); POP; STOP
        let mut store_log_call =
            vec![0x60, 0x01, 0x60, 0x00, 0x55, 0x60, 0x00, 0x60, 0x00, 0xa0, 0x60, 0x00, 0x60];
        store_log_call.extend([0x00, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0x73]);
        store_log_call.extend_from_slice(STORE.as_slice());
        store_log_call.extend([0x61, 0xff, 0xff, 0xf1, 0x50, 0x00]);
        db.insert_account_info(STORE_LOG_CALL, contract(store_log_call));
        // SSTORE(1, 2); STOP
        db.insert_account_info(STORE, contract(vec![0x60, 0x02, 0x60, 0x01, 0x55, 0x00]));
        // SSTORE(0, 1); REVERT(0, 0)
        db.insert_account_info(
            STORE_REVERT,
            contract(vec![0x60, 0x01, 0x60, 0x00, 0x55, 0x60, 0x00, 0x60, 0x00, 0xfd]),
        );
        db
    }

    fn chain_spec() -> Arc<OpChainSpec> {
        Arc::new(OpChainSpecBuilder::base_mainnet().isthmus_activated().build())
    }

    fn block() -> RecoveredBlock<alloy_consensus::Block<OpTransactionSigned>> {
        let transactions: Vec<OpTransactionSigned> = [STORE_LOG_CALL, STORE_REVERT, STORE]
            .into_iter()
            .enumerate()
            .map(|(i, to)| {
                TxDeposit {
                    source_hash: B256::with_last_byte(i as u8 + 1),
                    from: CALLER,
                    to: TxKind::Call(to),
                    gas_limit: 1_000_000,
                    ..Default::default()
                }
                .into()
            })
            .collect();
        let header = Header {
            parent_hash: B256::repeat_byte(3),
            beneficiary: Address::repeat_byte(0xfe),
            number: 1,
            timestamp: 1_000,
            gas_limit: 30_000_000,
            base_fee_per_gas: Some(1),
            mix_hash: B256::repeat_byte(2),
            parent_beacon_block_root: Some(B256::repeat_byte(1)),
            ..Default::default()
        };
        let senders = vec![CALLER; transactions.len()];
        RecoveredBlock::new_unhashed(
            alloy_consensus::Block {
                header,
                body: BlockBody { transactions, ..Default::default() },
            },
            senders,
        )
    }

    fn block_key(block: &RecoveredBlock<alloy_consensus::Block<OpTransactionSigned>>) -> B256 {
        trace_cache_key(
            block.header(),
            block.body().transactions.iter().map(|tx| *TxHashRef::tx_hash(tx)),
        )
    }

    fn debank_traces(traces: BlockTraces) -> String {
        let log_index = RefCell::new(0);
        let traces: Vec<_> = traces
            .into_iter()
            .map(|(hash, traces)| build_debank_traces(hash, traces, &log_index))
            .collect();
        serde_json::to_string(&traces).unwrap()
    }

    #[test]
    fn recorded_traces_match_replay() {
        let cache = Arc::new(DebankTraceCache::new(4));
        let config = DebankEvmConfig::new(OpEvmConfig::optimism(chain_spec()), Some(cache.clone()));
        let block = block();

        let mut state = State::builder().with_database(pre_state()).with_bundle_update().build();
        config
            .executor_for_block(&mut state, block.sealed_block())
            .unwrap()
            .execute_block(block.transactions_recovered())
            .unwrap();
        let recorded = cache.take(&block_key(&block)).expect("block traces are recorded");
        assert_eq!(recorded.len(), 3);

        // Replay the way `trace_debankBlock` does without the cache.
        let mut state = State::builder().with_database(pre_state()).with_bundle_update().build();
        config
            .executor_for_block(&mut state, block.sealed_block())
            .unwrap()
            .apply_pre_execution_changes()
            .unwrap();
        let evm_env = config.evm_env(block.header()).unwrap();
        let replayed: Vec<CallTraceArena> = config
            .evm_factory()
            .create_tracer(&mut state, evm_env, TracingInspector::new(debank_tracing_config()))
            .try_trace_many(block.transactions_recovered(), |mut ctx| {
                Ok::<_, eyre::Report>(ctx.take_inspector().into_traces())
            })
            .commit_last_tx()
            .collect::<Result<_, _>>()
            .unwrap();
        let replayed: BlockTraces = block
            .body()
            .transactions
            .iter()
            .map(|tx| *TxHashRef::tx_hash(tx))
            .zip(replayed)
            .collect();

        let recorded = debank_traces(recorded);
        assert_eq!(recorded, debank_traces(replayed));
        assert!(recorded.contains("\"storage_change\":true"));
        assert!(recorded.contains("Reverted"));
    }

    #[test]
    fn evm_with_external_inspector_records_nothing() {
        let cache = Arc::new(DebankTraceCache::new(4));
        let config = DebankEvmConfig::new(OpEvmConfig::optimism(chain_spec()), Some(cache.clone()));
        let block = block();

        let mut state = State::builder().with_database(pre_state()).with_bundle_update().build();
        let evm = config.evm_factory().create_evm_with_inspector(
            &mut state,
            config.evm_env(block.header()).unwrap(),
            NoOpInspector,
        );
        let ctx = config.context_for_block(block.sealed_block()).unwrap();
        config.create_executor(evm, ctx).execute_block(block.transactions_recovered()).unwrap();

        assert!(cache.take(&block_key(&block)).is_none());
    }

    #[test]
    fn cache_evicts_oldest_block() {
        let cache = DebankTraceCache::new(2);
        let [a, b, c] = [1, 2, 3].map(B256::with_last_byte);
        cache.insert(a, Vec::new());
        cache.insert(b, Vec::new());
        cache.insert(a, Vec::new());
        cache.insert(c, Vec::new());

        assert!(cache.take(&a).is_none());
        assert!(cache.take(&b).is_some());
        assert!(cache.take(&b).is_none());
        assert!(cache.take(&c).is_some());
    }
}
