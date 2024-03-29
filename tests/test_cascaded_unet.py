import pytest
import torch
from monai.networks import eval_mode
from monai.networks.utils import normal_init

from swin_unetr.cascaded_unet import CascadedUNet


def compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module) -> bool:
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismatch found at", key_item_1[0])
            else:
                raise Exception
    return models_differ == 0


def test_CascadedUNet_serialize(tmp_path):
    m = CascadedUNet(1, [2], 2, include_background_channel=False)
    assert len(m.feature_nets) == 1

    nets = [m.net] + list(m.feature_nets.modules())
    for net_i in nets:
        for sm in net_i.modules():
            normal_init(sm)

    torch.save(m, tmp_path / "foo.pt")

    m2 = torch.load(tmp_path / "foo.pt")
    assert isinstance(m2, CascadedUNet)

    assert compare_models(m, m2)
    assert compare_models(m.feature_nets[0], m2.feature_nets[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_CascadedUNet_device():
    device = torch.device("cuda:0")
    m = CascadedUNet(1, [2], 2, include_background_channel=False)
    m.to(device)

    assert next(m.net.parameters()).is_cuda
    assert next(m.feature_nets[0].parameters()).is_cuda


def test_CascadedUNet_eval():
    input_shape = [1, 1, 96, 96, 96]
    expected_shape = [1, 2, 96, 96, 96]
    net = CascadedUNet(1, [3], 2, include_background_channel=False)
    with eval_mode(net):
        result = net(torch.randn(input_shape))
        assert list(result.shape) == expected_shape
