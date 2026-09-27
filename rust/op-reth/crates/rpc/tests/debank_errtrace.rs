//! Regression tests for failed-parent `DeBank` traces.

use alloy_primitives::{Address, B256 as H256};
use reth_optimism_rpc::debank::{DebankTrace, build_debank_traces};
use revm::interpreter::InstructionResult;
use revm_inspectors::tracing::{
    CallTraceArena,
    types::{CallKind, CallLog, CallTrace, CallTraceNode, TraceMemberOrder},
};
use std::cell::RefCell;

const PARENT_CALL_FAILED_ERROR: &str = "parent call failed";

fn addr(byte: u8) -> Address {
    Address::repeat_byte(byte)
}

fn call_trace_node(
    idx: usize,
    parent: Option<usize>,
    address: Address,
    success: bool,
    status: InstructionResult,
    children: Vec<usize>,
) -> CallTraceNode {
    let ordering = (0..children.len()).map(TraceMemberOrder::Call).collect();
    CallTraceNode {
        parent,
        children,
        idx,
        trace: CallTrace {
            success,
            address,
            kind: CallKind::Call,
            status: Some(status),
            ..Default::default()
        },
        ordering,
        ..Default::default()
    }
}

fn call_trace_arena(nodes: Vec<CallTraceNode>) -> CallTraceArena {
    let mut arena = CallTraceArena::default();
    *arena.nodes_mut() = nodes;
    arena
}

fn trace_for_address(traces: &[DebankTrace], address: Address) -> &DebankTrace {
    traces.iter().find(|trace| trace.to_addr == address).unwrap()
}

fn with_log(mut node: CallTraceNode) -> CallTraceNode {
    node.logs.push(CallLog::default());
    node.ordering.push(TraceMemberOrder::Log(0));
    node
}

#[test]
fn successful_descendants_of_failed_call_are_error_traces() {
    let arena = call_trace_arena(vec![
        call_trace_node(0, None, addr(1), true, InstructionResult::Stop, vec![1, 4]),
        call_trace_node(1, Some(0), addr(2), false, InstructionResult::Revert, vec![2, 3]),
        call_trace_node(2, Some(1), addr(3), true, InstructionResult::Stop, vec![]),
        call_trace_node(3, Some(1), addr(4), false, InstructionResult::OutOfGas, vec![]),
        call_trace_node(4, Some(0), addr(5), true, InstructionResult::Stop, vec![]),
    ]);

    let (traces, error_traces, _, _) =
        build_debank_traces(H256::repeat_byte(0xaa), arena, &RefCell::new(0));

    assert_eq!(traces.len(), 2);
    assert_eq!(error_traces.len(), 3);
    assert_eq!(trace_for_address(&error_traces, addr(2)).error, "Reverted");
    assert_eq!(trace_for_address(&error_traces, addr(3)).error, PARENT_CALL_FAILED_ERROR);
    assert_eq!(trace_for_address(&error_traces, addr(4)).error, "Out of gas");
}

#[test]
fn failed_parent_marks_selfdestruct_and_uses_nested_trace_address() {
    let mut selfdestruct =
        call_trace_node(1, Some(0), addr(2), true, InstructionResult::SelfDestruct, vec![]);
    selfdestruct.trace.selfdestruct_address = Some(addr(3));
    selfdestruct.trace.selfdestruct_refund_target = Some(addr(4));
    let arena = call_trace_arena(vec![
        call_trace_node(0, None, addr(1), false, InstructionResult::Revert, vec![1]),
        selfdestruct,
    ]);

    let (traces, error_traces, _, _) =
        build_debank_traces(H256::repeat_byte(0xaa), arena, &RefCell::new(0));

    assert!(traces.is_empty());
    assert_eq!(error_traces.len(), 3);
    let child = trace_for_address(&error_traces, addr(2));
    assert_eq!(child.error, PARENT_CALL_FAILED_ERROR);
    assert_eq!(child.trace_address, vec![0]);
    let selfdestruct = trace_for_address(&error_traces, addr(4));
    assert_eq!(selfdestruct.error, PARENT_CALL_FAILED_ERROR);
    assert_eq!(selfdestruct.trace_address, vec![0, 0]);
}

#[test]
fn root_selfdestruct_uses_first_child_trace_address() {
    let mut root = call_trace_node(0, None, addr(1), true, InstructionResult::SelfDestruct, vec![]);
    root.trace.selfdestruct_address = Some(addr(2));
    root.trace.selfdestruct_refund_target = Some(addr(3));

    let (traces, error_traces, _, _) = build_debank_traces(
        H256::repeat_byte(0xaa),
        call_trace_arena(vec![root]),
        &RefCell::new(0),
    );

    assert!(error_traces.is_empty());
    assert_eq!(traces.len(), 2);
    let selfdestruct = trace_for_address(&traces, addr(3));
    assert_eq!(selfdestruct.trace_address, vec![0]);
}

#[test]
fn selfdestruct_follows_existing_child_trace_addresses() {
    let mut selfdestruct =
        call_trace_node(1, Some(0), addr(2), true, InstructionResult::SelfDestruct, vec![2]);
    selfdestruct.trace.selfdestruct_address = Some(addr(3));
    selfdestruct.trace.selfdestruct_refund_target = Some(addr(4));
    let arena = call_trace_arena(vec![
        call_trace_node(0, None, addr(1), true, InstructionResult::Stop, vec![1]),
        selfdestruct,
        call_trace_node(2, Some(1), addr(5), true, InstructionResult::Stop, vec![]),
    ]);

    let (traces, error_traces, _, _) =
        build_debank_traces(H256::repeat_byte(0xaa), arena, &RefCell::new(0));

    assert!(error_traces.is_empty());
    assert_eq!(trace_for_address(&traces, addr(5)).trace_address, vec![0, 0]);
    assert_eq!(trace_for_address(&traces, addr(4)).trace_address, vec![0, 1]);
}

#[test]
fn error_event_index_stays_zero_after_successful_event() {
    let arena = call_trace_arena(vec![
        call_trace_node(0, None, addr(1), true, InstructionResult::Stop, vec![1, 2]),
        with_log(call_trace_node(1, Some(0), addr(2), true, InstructionResult::Stop, vec![])),
        with_log(call_trace_node(2, Some(0), addr(3), false, InstructionResult::Revert, vec![])),
    ]);

    let (_, _, events, error_events) =
        build_debank_traces(H256::repeat_byte(0xaa), arena, &RefCell::new(0));

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].idx, 0);
    assert_eq!(error_events.len(), 1);
    assert_eq!(error_events[0].idx, 0);
}
