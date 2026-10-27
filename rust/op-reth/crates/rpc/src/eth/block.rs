//! Loads and formats OP block RPC response.

use crate::{OpEthApi, OpEthApiError, eth::RpcNodeCore};
use alloy_eips::BlockId;
use reth_chainspec::{ChainSpecProvider, EthChainSpec};
use reth_primitives_traits::RecoveredBlock;
use reth_rpc_eth_api::{
    FromEthApiError, FromEvmError, RpcConvert,
    helpers::{EthBlocks, LoadBlock, LoadPendingBlock, SpawnBlocking},
    node::RpcNodeCoreExt,
};
use reth_rpc_eth_types::EthApiError;
use reth_storage_api::{BlockIdReader, BlockReader};
use std::sync::Arc;

impl<N, Rpc> EthBlocks for OpEthApi<N, Rpc>
where
    N: RpcNodeCore,
    OpEthApiError: FromEvmError<N::Evm>,
    Rpc: RpcConvert<Primitives = N::Primitives, Error = OpEthApiError>,
{
}

impl<N, Rpc> LoadBlock for OpEthApi<N, Rpc>
where
    N: RpcNodeCore,
    OpEthApiError: FromEvmError<N::Evm>,
    Rpc: RpcConvert<Primitives = N::Primitives, Error = OpEthApiError>,
{
    async fn recovered_block(
        &self,
        block_id: BlockId,
    ) -> Result<Option<Arc<RecoveredBlock<<Self::Provider as BlockReader>::Block>>>, Self::Error>
    {
        // Preserve reth's pending-block handling and cache path for ordinary blocks.
        if block_id.is_pending() {
            if self.pending_block_kind().is_none() {
                return Ok(None);
            }
            if let Some(pending) =
                self.provider().pending_block().map_err(Self::Error::from_eth_err)?
            {
                return Ok(Some(Arc::new(pending)));
            }
            return Ok(self.local_pending_block().await?.map(|pending| pending.block));
        }

        let Some(block_hash) =
            self.provider().block_hash_for_id(block_id).map_err(Self::Error::from_eth_err)?
        else {
            return Ok(None);
        };

        if block_hash == self.provider().chain_spec().genesis_hash() {
            // OP-node checks genesis on every startup. Reth's single-block reader rejects it
            // below the pruned static-file boundary even though its header/body are still stored.
            // The native range reader loads that persisted block without fabricating a body.
            return self
                .spawn_blocking_io(move |this| {
                    let block = this
                        .provider()
                        .recovered_block_range(0..=0)
                        .map_err(Self::Error::from_eth_err)?
                        .pop();
                    if let Some(block) = &block &&
                        block.hash() != block_hash
                    {
                        return Err(EthApiError::InternalEthError.into());
                    }
                    Ok(block.map(Arc::new))
                })
                .await;
        }

        self.cache().get_recovered_block(block_hash).await.map_err(Self::Error::from_eth_err)
    }
}
