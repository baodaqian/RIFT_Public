"""Count-bound collection observer, separate from frozen B787 evidence gates."""
from .b7873200_adaptive_fullscale import AdaptiveFullScaleObserver, FULLSCALE_EPOCHS
import torch

from .rift_dataset import collection_contract, PARENT_NUM_TRAIN


class SelectedAntennaObserver(AdaptiveFullScaleObserver):
    def _render_item(self, item):
        from train import apply_occlusion, cc, get_kvector
        from .range_operator import range_forward_operator
        freqs, phi, theta, magnitude, phase, rx, tx = item
        if freqs.numel() != self.num_freq_selected:
            raise ValueError('Collection observer requires every source frequency')
        if magnitude.shape[-1] != len(tx) * len(rx) or phase.shape != magnitude.shape:
            raise ValueError('Observer measurement shape differs from selected geometry')
        freqs, rx, tx = freqs.to(self.device), rx.to(self.device), tx.to(self.device)
        xyz, weights = self.model.active_scatterers(theta.unsqueeze(0).to(self.device), phi.unsqueeze(0).to(self.device))
        weights = apply_occlusion(self.model, weights, rx, tx, self.occlusion)
        prediction = range_forward_operator(freqs, get_kvector(freqs, cc), rx, tx, xyz, weights,
            phase_sign=self.phase_sign, freq_indices=torch.arange(len(freqs), device=self.device),
            compute_dtype=self.compute_dtype, **self.op_kwargs)
        return self.gain(prediction) if self.gain is not None else prediction


def observer_type(contract):
    selected = collection_contract(contract)
    if selected is None:
        raise ValueError("Collection observer requires a validated object/role contract")
    count = len(selected["role_ids"]["train"])
    acquisition = selected.get("antenna_selection")
    if count == PARENT_NUM_TRAIN and not acquisition:
        return AdaptiveFullScaleObserver
    from .antenna_selection import acquisition_label
    suffix = "_" + acquisition_label(acquisition) if acquisition else ""
    return type("AdaptiveCollectionSubsetObserver", (SelectedAntennaObserver if acquisition else AdaptiveFullScaleObserver,), {
        "schema": f"rift_collection_adaptive_train{count}{suffix}_observer_v1",
        "num_train": count,
        "expected_updates": count * FULLSCALE_EPOCHS,
    })


def execution_label(num_train):
    return f"rift_dataset_adaptive_train{num_train}_v1"
