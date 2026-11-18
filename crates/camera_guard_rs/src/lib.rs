//! `camera_guard_rs`: Real-time ADAS camera reliability monitor and safety guard.

pub mod ambient;
pub mod guard;
pub mod metrics;
pub mod types;

pub use ambient::estimate_ambient;
pub use guard::{classify_system_state, compute_dynamic_sensor_weight, evaluate_rgb_frame, score_from_metrics};
pub use metrics::compute_optical_metrics;
pub use types::{AmbientEstimate, AmbientRegime, GuardConfig, GuardResult, OpticalMetrics, SystemState};

#[cfg(feature = "python")]
mod py_bindings {
    use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::guard::evaluate_rgb_frame;
    use crate::metrics::compute_optical_metrics;
    use crate::types::{GuardConfig, GuardResult};

    /// Python-exposed GuardResult mirroring `scripts/camera_guard.py::GuardResult`.
    #[pyclass(name = "GuardResult", get_all)]
    #[allow(non_snake_case)]
    #[derive(Clone)]
    pub struct PyGuardResult {
        pub health_score: f64,
        pub system_state: String,
        pub camera_weight: f64,
        pub early_warning_flag: bool,
        pub B: f64,
        pub B_norm: f64,
        pub S_pct: f64,
        pub H_bit: f64,
        pub ambient_regime: String,
        pub daylight_factor: f64,
        pub B_baseline_dynamic: f64,
        pub mean_luma: f64,
        pub p95_luma: f64,
        pub dark_ratio_pct: f64,
        pub pred_health: Option<f64>,
        pub delta_discordance: f64,
        pub discordance_triggered: bool,
        pub discordance_factor: f64,
        pub quality_penalty: f64,
    }

    impl From<GuardResult> for PyGuardResult {
        fn from(r: GuardResult) -> Self {
            Self {
                health_score: r.health_score,
                system_state: r.system_state.as_str().to_string(),
                camera_weight: r.camera_weight,
                early_warning_flag: r.early_warning_flag,
                B: r.b,
                B_norm: r.b_norm,
                S_pct: r.s_pct,
                H_bit: r.h_bit,
                ambient_regime: r.ambient_regime.as_str().to_string(),
                daylight_factor: r.daylight_factor,
                B_baseline_dynamic: r.b_baseline_dynamic,
                mean_luma: r.mean_luma,
                p95_luma: r.p95_luma,
                dark_ratio_pct: r.dark_ratio_pct,
                pred_health: r.pred_health,
                delta_discordance: r.delta_discordance,
                discordance_triggered: r.discordance_triggered,
                discordance_factor: r.discordance_factor,
                quality_penalty: r.quality_penalty,
            }
        }
    }

    #[pymethods]
    impl PyGuardResult {
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("health_score", self.health_score)?;
            d.set_item("system_state", &self.system_state)?;
            d.set_item("camera_weight", self.camera_weight)?;
            d.set_item("early_warning_flag", self.early_warning_flag)?;
            d.set_item("B", self.B)?;
            d.set_item("B_norm", self.B_norm)?;
            d.set_item("S_pct", self.S_pct)?;
            d.set_item("H_bit", self.H_bit)?;
            d.set_item("ambient_regime", &self.ambient_regime)?;
            d.set_item("daylight_factor", self.daylight_factor)?;
            d.set_item("B_baseline_dynamic", self.B_baseline_dynamic)?;
            d.set_item("mean_luma", self.mean_luma)?;
            d.set_item("p95_luma", self.p95_luma)?;
            d.set_item("dark_ratio_pct", self.dark_ratio_pct)?;
            d.set_item("pred_health", self.pred_health)?;
            d.set_item("delta_discordance", self.delta_discordance)?;
            d.set_item("discordance_triggered", self.discordance_triggered)?;
            d.set_item("discordance_factor", self.discordance_factor)?;
            d.set_item("quality_penalty", self.quality_penalty)?;
            Ok(d)
        }

        fn __repr__(&self) -> String {
            format!(
                "GuardResult(health_score={:.6}, system_state='{}', camera_weight={:.4}, B={:.4}, early_warning={})",
                self.health_score, self.system_state, self.camera_weight, self.B, self.early_warning_flag
            )
        }
    }

    /// Zero-copy evaluation of a contiguous HxWx3 uint8 RGB NumPy array.
    #[pyfunction]
    #[pyo3(signature = (image_np, pred_health=None, adaptive_day_night=true))]
    fn evaluate_frame(
        py: Python<'_>,
        image_np: PyReadonlyArray3<'_, u8>,
        pred_health: Option<f64>,
        adaptive_day_night: bool,
    ) -> PyResult<PyGuardResult> {
        let shape = image_np.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected HxWx3 RGB uint8 array, got shape {:?}",
                shape
            )));
        }
        let (height, width) = (shape[0], shape[1]);
        let cfg = GuardConfig::default();

        if let Ok(slice) = image_np.as_slice() {
            let res = py
                .allow_threads(|| evaluate_rgb_frame(slice, height, width, pred_health, adaptive_day_night, &cfg))
                .map_err(PyValueError::new_err)?;
            Ok(res.into())
        } else {
            let owned: Vec<u8> = image_np.as_array().iter().copied().collect();
            let res = py
                .allow_threads(|| evaluate_rgb_frame(&owned, height, width, pred_health, adaptive_day_night, &cfg))
                .map_err(PyValueError::new_err)?;
            Ok(res.into())
        }
    }

    /// Zero-copy computation of handcrafted optical metrics (B, S_pct, H_bit, mean_luma, p95_luma, dark_ratio_pct).
    #[pyfunction]
    fn compute_metrics<'py>(py: Python<'py>, image_np: PyReadonlyArray3<'_, u8>) -> PyResult<Bound<'py, PyDict>> {
        let shape = image_np.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected HxWx3 RGB uint8 array, got shape {:?}",
                shape
            )));
        }
        let (height, width) = (shape[0], shape[1]);
        let m = if let Ok(slice) = image_np.as_slice() {
            py.allow_threads(|| compute_optical_metrics(slice, height, width))
                .map_err(PyValueError::new_err)?
        } else {
            let owned: Vec<u8> = image_np.as_array().iter().copied().collect();
            py.allow_threads(|| compute_optical_metrics(&owned, height, width))
                .map_err(PyValueError::new_err)?
        };
        let d = PyDict::new(py);
        d.set_item("B", m.b)?;
        d.set_item("S_pct", m.s_pct)?;
        d.set_item("H_bit", m.h_bit)?;
        d.set_item("mean_luma", m.mean_luma)?;
        d.set_item("p95_luma", m.p95_luma)?;
        d.set_item("dark_ratio_pct", m.dark_ratio_pct)?;
        Ok(d)
    }

    #[pymodule]
    fn camera_guard_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyGuardResult>()?;
        m.add_function(wrap_pyfunction!(evaluate_frame, m)?)?;
        m.add_function(wrap_pyfunction!(compute_metrics, m)?)?;
        Ok(())
    }
}
