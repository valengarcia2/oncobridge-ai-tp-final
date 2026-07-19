"""
Tests de las métricas de Componente 2. Todo sintético y sin LLM -- corre
siempre, instantáneo.
"""

from oncobridge.evaluation.metrics_component2 import (
    aggregate_component2_metrics,
    classification_is_applicable,
    expected_classification,
    findings_confusion_bucket,
    segmentation_zone_iou_proxy,
)
from oncobridge.schemas.component1_io import TokenUsage
from oncobridge.schemas.component2_io import Component2Output, RegionOfInterest, Segmentation
from oncobridge.schemas.evaluation import ExpectedOutput
from oncobridge.schemas.ground_truth import (
    GroundTruthEntry,
    ImagingLocation,
    ObjectiveData,
    RadiologistGuidance,
    SubjectiveData,
)


def _gt_entry(gt_id, icd_10, priority_zones, anatomical_landmarks) -> GroundTruthEntry:
    return GroundTruthEntry(
        gt_id=gt_id,
        icd_10=icd_10,
        icd_10_description="test",
        objective_data=ObjectiveData(),
        subjective_data=SubjectiveData(),
        radiologist_guidance=RadiologistGuidance(
            imaging_location=ImagingLocation(
                anatomical_landmarks=anatomical_landmarks,
                priority_zones=priority_zones,
            )
        ),
        base_probability=0.7,
        urgency_level="alta",
    )


def _c2_output(classification="sospechoso", confidence=0.9, roi_location="pulmon derecho") -> Component2Output:
    return Component2Output(
        patient_id="PAT-TEST",
        segmentation=Segmentation(
            regions_of_interest=[
                RegionOfInterest(
                    id="ROI-01", location=roi_location, size_mm=20.0, shape="test", margins="test", density="test"
                )
            ]
        ),
        findings="test",
        classification=classification,
        confidence=confidence,
        final_recommendation="test",
        next_steps=[],
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200, model="test"),
    )


def _expected(case_id="case_test", correct_gt_ids=None) -> ExpectedOutput:
    return ExpectedOutput(
        case_id=case_id,
        correct_gt_ids=correct_gt_ids or [],
        acceptable_secondary_gt_ids=[],
        imaging_needed_ground_truth=True,
        urgency_ground_truth="alta",
        specialist_decision="DERIVAR_A_IMAGEN",
        conclusive_ground_truth=True,
        difficulty="facil",
    )


def test_classification_is_applicable_requires_correct_gt_ids():
    assert classification_is_applicable(_expected(correct_gt_ids=["GT-A"])) is True
    assert classification_is_applicable(_expected(correct_gt_ids=[])) is False


def test_expected_classification_derived_from_icd10():
    gt_index = {"GT-MALIGNO": _gt_entry("GT-MALIGNO", "C34.1", ["pulmon"], "pulmon")}
    expected = _expected(correct_gt_ids=["GT-MALIGNO"])
    assert expected_classification(expected, gt_index) == "sospechoso"


def test_expected_classification_none_without_correct_gt_ids():
    assert expected_classification(_expected(correct_gt_ids=[]), {}) is None


def test_findings_bucket_true_positive():
    output = _c2_output(classification="sospechoso")
    bucket = findings_confusion_bucket(output, expected_class="sospechoso")
    assert bucket.true_positive is True


def test_findings_bucket_false_negative():
    output = _c2_output(classification="benigno")
    bucket = findings_confusion_bucket(output, expected_class="sospechoso")
    assert bucket.false_negative is True


def test_findings_bucket_false_positive():
    output = _c2_output(classification="sospechoso")
    bucket = findings_confusion_bucket(output, expected_class="benigno")
    assert bucket.false_positive is True


def test_findings_bucket_true_negative():
    output = _c2_output(classification="benigno")
    bucket = findings_confusion_bucket(output, expected_class="benigno")
    assert bucket.true_negative is True


def test_segmentation_iou_proxy_full_overlap():
    entry = _gt_entry("GT-A", "C34.1", ["nodulo pulmonar derecho"], "pulmon derecho")
    output = _c2_output(roi_location="nodulo pulmonar derecho, pulmon derecho")
    iou = segmentation_zone_iou_proxy(output, entry)
    assert iou == 1.0


def test_segmentation_iou_proxy_no_overlap():
    entry = _gt_entry("GT-A", "C34.1", ["nodulo pulmonar derecho"], "pulmon derecho")
    output = _c2_output(roi_location="masa hepatica izquierda")
    iou = segmentation_zone_iou_proxy(output, entry)
    assert iou == 0.0


def test_segmentation_iou_proxy_partial_overlap_is_between_0_and_1():
    entry = _gt_entry("GT-A", "C34.1", ["nodulo pulmonar derecho"], "pulmon derecho hilio")
    output = _c2_output(roi_location="pulmon derecho")
    iou = segmentation_zone_iou_proxy(output, entry)
    assert 0.0 < iou < 1.0


def test_aggregate_metrics_over_synthetic_cases():
    gt_index = {
        "GT-MALIGNO": _gt_entry("GT-MALIGNO", "C34.1", ["nodulo pulmonar"], "pulmon derecho"),
        "GT-BENIGNO": _gt_entry("GT-BENIGNO", "D30.0", ["quiste renal"], "rinon"),
    }

    cases = [
        # TP: maligno real, sistema dice sospechoso, zonas coinciden
        (
            _c2_output(classification="sospechoso", confidence=0.9, roi_location="nodulo pulmonar derecho"),
            _expected(case_id="case_a", correct_gt_ids=["GT-MALIGNO"]),
        ),
        # TN: benigno real, sistema dice benigno
        (
            _c2_output(classification="benigno", confidence=0.4, roi_location="quiste renal"),
            _expected(case_id="case_b", correct_gt_ids=["GT-BENIGNO"]),
        ),
        # No aplicable: sin correct_gt_ids (tipo case_109) -- no debe entrar en sensibilidad/especificidad
        (
            _c2_output(classification="sospechoso", confidence=0.6, roi_location="masa retroperitoneal"),
            _expected(case_id="case_109_like", correct_gt_ids=[]),
        ),
    ]

    report = aggregate_component2_metrics(cases, gt_index)

    assert report.n_cases == 3
    assert report.n_casos_clasificacion_aplicable == 2
    assert report.sensibilidad_hallazgos == 1.0
    assert report.especificidad_hallazgos == 1.0
    assert 0.0 <= report.segmentacion_iou_proxy_promedio <= 1.0
