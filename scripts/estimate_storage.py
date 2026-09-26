#!/usr/bin/env python3
"""Checkpoint/cache budgets for current RIFT, SpINR and SE recipes (2026-09-20).

Pure arithmetic; no data access, Torch, fitting or scheduler operations.
MiB/GiB are binary. Input archives, logs and geometry exports are additional.
Model payloads were counted from CPU state_dict tensors, adding two Adam moments
per parameter; gradients and derived quadrature arrays are not checkpointed.
RIFT assumes capacity=262144 and SH degree=3; one GOTCHA HH head.

Source contracts:
  train.py: save_run_checkpoint, best/latest/final; rift/gotcha_training.py: run.
  train_spinr_style.py: _checkpoint_state, _CHECKPOINT_FILENAMES.
  rift/spinr_gotcha_training.py: PulsePlan.coverage, state, history.json.
  rift/sugavanam_ertin_paper_workflow.py: run, extract_cloud, _save.

Native SpINR history sizes below came from CPU synthetic serialization of 1500
coverage records using the registered metadata-only PulsePlan (236826 pulses,
2000 sectors). Response reads were guarded off. Validation metrics/other small
metadata get an additional 1 MiB/checkpoint and 1 MiB/history JSON allowance.
SE scalar-history sizes use synthetic full-budget records (10800 + 5000), not
fitting. Its cloud upper bound is 64 bytes/grid point and iso tensors 25*2048.
These are rounded planning budgets, not measured production artifacts or hard
bounds on arbitrarily changed recipes. SE numbers are conditional on its gates.
"""
import argparse
import json
from math import ceil

MIB = 2**20
GIB = 2**30
META = MIB
SPINR_TENSORS = 42_799_692
SE_SDF_TENSORS = 22_333_560
NATIVE_HISTORY = 49_923_104
NATIVE_HISTORY_JSON = 190_063_080
SE_HISTORY = 2_645_693 + 499_149


def rows():
    result = []

    def add(model, dataset, payload, files, *, history_json=0, selected=0):
        checkpoint = ceil((payload + META) / MIB) * MIB
        history = ceil(history_json / MIB) * MIB
        selected = ceil(selected / MIB) * MIB
        retained = files * checkpoint + history + selected
        result.append(dict(
            model=model, dataset=dataset, response_disk_cache_bytes=0,
            checkpoint_tensor_and_history_bytes=payload,
            checkpoint_budget_mib=checkpoint // MIB, retained_checkpoints=files,
            history_json_budget_mib=history // MIB,
            selected_stage1_budget_mib=selected // MIB,
            retained_budget_mib=retained // MIB,
            atomic_peak_budget_mib=(retained + max(checkpoint, history)) // MIB,
            conditional=model == 'SE'))

    # Collection gain + Adam adds 24 bytes to the adaptive-scene inventory.
    add('RIFT', 'Collection', 138_674_354 + 24, 3)
    add('RIFT', 'GOTCHA', 139_460_812, 3)
    add('SpINR', 'Collection', SPINR_TENSORS, 4, history_json=MIB)
    add('SpINR', 'GOTCHA', SPINR_TENSORS + NATIVE_HISTORY, 3,
        history_json=NATIVE_HISTORY_JSON + MIB)
    for dataset, grid in [('Collection', 7), ('GOTCHA', 42)]:
        bank = 72 * grid**3 * 16  # complex128 per subaperture/grid point
        cloud = 64 * grid**3     # all grid points retained: upper payload bound
        full = 2*bank + SE_SDF_TENSORS + SE_HISTORY + cloud + 25*2048
        add('SE', dataset, full, 2, selected=bank + cloud + META)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = rows()
    auxiliary = dict(
        collection_rift_or_spinr_response_ram_gib=4200*256*600*8/GIB,
        spinr_collection_quadrature_device_mib=(96*2)**3*32/MIB,
        spinr_gotcha_quadrature_device_mib=(48*2)**3*32/MIB,
        spinr_gotcha_pulse_plan_host_mib=7_594_440/MIB,
        native_roi_basis_device_upper_mib=64*434*433*16/MIB,
        se_collection_two_cpu_field_banks_mib=2*72*7**3*16/MIB,
        se_gotcha_two_cpu_field_banks_mib=2*72*42**3*16/MIB)
    if args.json:
        print(json.dumps(dict(checkpoints=result, memory=auxiliary), indent=2))
        return
    print('| Method / dataset | Checkpoint MiB | Retained files | Retained MiB | Atomic peak MiB |')
    print('| --- | ---: | ---: | ---: | ---: |')
    for r in result:
        print(f"| {r['model']} / {r['dataset']} | {r['checkpoint_budget_mib']} | "
              f"{r['retained_checkpoints']} | {r['retained_budget_mib']} | {r['atomic_peak_budget_mib']} |")
    print('\nRAM/device caches (not additional disk):')
    print(json.dumps(auxiliary, indent=2))


if __name__ == '__main__':
    main()
