from toploc.C.csrc.utils import get_fp_parts
import torch
import time
import pytest
import tempfile
from toploc.utils import sha256sum


@pytest.mark.parametrize(
    "tensor_shape",
    [
        (10_000,),
        (1000, 3),
        (1, 3, 5, 8, 1),
    ],
)
def test_get_fp32_parts(tensor_shape: tuple[int, ...]) -> None:
    a = torch.randn(tensor_shape)
    start_time = time.time()
    exps, mantissas = get_fp_parts(a)
    new_time = time.time() - start_time

    ELEMENT_SIZE = 4

    def get_exp_bits(fp32_str: str) -> int:
        return int(fp32_str[1:9], 2)

    def get_mant_bits(fp32_str: str) -> int:
        return int(fp32_str[9:], 2)

    def py_get_fp_parts(tensor: torch.FloatTensor) -> tuple[list[int], list[int]]:
        """
        Given a tensor of floats, return the exponent and mantissa bits of each float.

        Args:
            tensor (torch.FloatTensor): A tensor of floats.

        Returns:
            tuple[list[int], list[int]]: A tuple containing the exponent and mantissa bits of each float.
        """
        temp = None
        bit_repr = []
        for i, e in enumerate(tensor.untyped_storage()):
            if i % ELEMENT_SIZE == 0:
                if temp is not None:
                    bit_repr.append("".join(temp[::-1]))
                temp = [bin(e)[2:].zfill(8)]
            else:
                temp.append(bin(e)[2:].zfill(8))
        bit_repr.append("".join(temp[::-1]))

        prefill_exps = [get_exp_bits(i) for i in bit_repr]
        prefill_mants = [get_mant_bits(i) for i in bit_repr]
        return prefill_exps, prefill_mants

    start_time = time.time()
    ref_exps, ref_mants = py_get_fp_parts(a)
    old_time = time.time() - start_time

    assert exps == ref_exps
    assert mantissas == ref_mants
    assert new_time < old_time


@pytest.mark.parametrize(
    "tensor_shape",
    [
        (10_000,),
        (1000, 3),
        (1, 3, 5, 8, 1),
    ],
)
def test_get_bf16_parts(tensor_shape: tuple[int, ...]) -> None:
    a = torch.randn(tensor_shape, dtype=torch.bfloat16)
    start_time = time.time()
    exps, mantissas = get_fp_parts(a)
    new_time = time.time() - start_time

    ELEMENT_SIZE = 2

    def get_exp_bits(fp32_str: str) -> int:
        return int(fp32_str[1:9], 2)

    def get_mant_bits(fp32_str: str) -> int:
        return int(fp32_str[9:], 2)

    def py_get_fp_parts(tensor: torch.FloatTensor) -> tuple[list[int], list[int]]:
        """
        Given a tensor of floats, return the exponent and mantissa bits of each float.

        Args:
            tensor (torch.FloatTensor): A tensor of floats.

        Returns:
            tuple[list[int], list[int]]: A tuple containing the exponent and mantissa bits of each float.
        """
        temp = None
        bit_repr = []
        for i, e in enumerate(tensor.untyped_storage()):
            if i % ELEMENT_SIZE == 0:
                if temp is not None:
                    bit_repr.append("".join(temp[::-1]))
                temp = [bin(e)[2:].zfill(8)]
            else:
                temp.append(bin(e)[2:].zfill(8))
        bit_repr.append("".join(temp[::-1]))

        prefill_exps = [get_exp_bits(i) for i in bit_repr]
        prefill_mants = [get_mant_bits(i) for i in bit_repr]
        return prefill_exps, prefill_mants

    start_time = time.time()
    ref_exps, ref_mants = py_get_fp_parts(a)
    old_time = time.time() - start_time

    assert exps == ref_exps
    assert mantissas == ref_mants
    assert new_time < old_time


def test_sha256sum():
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"Hello, world!" * 1000)
        f.flush()
        assert (
            sha256sum(f.name)
            == "a8f764e70df94be2c911fb51b3d0c56c03882078dbdb215de8b7bd0374b0fb10"
        )
