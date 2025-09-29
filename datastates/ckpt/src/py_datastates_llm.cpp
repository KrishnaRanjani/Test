#include <torch/extension.h>
#include "engine.hpp"
#include <pybind11/iostream.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        DataStates-LLM
        -----------------------
        .. currentmodule:: datastates
        .. autosummary::
           :toctree: _generate
           ckpt_tensor
           restore_tensor
           wait
           shutdown
           should_force_full_checkpoint
           increment_checkpoint_epoch
           is_warmup_complete_status
           get_current_epoch
           set_skip_checkpoint
           reset_checkpoint_skip
           get_skipped_checkpoint_count
    )pbdoc";

    py::class_<datastates_llm_t>(m, "handle")
        .def(py::init<const size_t, int, int>(), 
             py::arg("host_cache_size"), py::arg("gpu_id"), py::arg("rank") = -1)
        .def("ckpt_tensor", &datastates_llm_t::ckpt_tensor, py::call_guard<py::gil_scoped_release>())
        .def("restore_tensor", &datastates_llm_t::restore_tensor, py::call_guard<py::gil_scoped_release>())
        .def("wait", &datastates_llm_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &datastates_llm_t::shutdown)
        .def("should_force_full_checkpoint", &datastates_llm_t::should_force_full_checkpoint)
        .def("increment_checkpoint_epoch", &datastates_llm_t::increment_checkpoint_epoch)
        .def("is_warmup_complete_status", &datastates_llm_t::is_warmup_complete_status)
        .def("get_current_epoch", &datastates_llm_t::get_current_epoch)
        .def("set_skip_checkpoint", &datastates_llm_t::set_skip_checkpoint)
        .def("reset_checkpoint_skip", &datastates_llm_t::reset_checkpoint_skip)
        .def("get_skipped_checkpoint_count", &datastates_llm_t::get_skipped_checkpoint_count);
}