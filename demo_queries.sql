-- ============================================================================
-- Demo queries for ehr.duckdb
--
--   duckdb ehr.duckdb -f demo_queries.sql        (all of them)
--   duckdb ehr.duckdb                            (interactive, paste one)
--
-- Every query runs against the views, not the raw tables. See README.md for
-- why the views exist.
--
-- NOTE: vitals are identical across both visits for all 53 two-encounter
-- patients, so nothing here trends a vital sign. Labs do vary -- query 6
-- trends those instead.
-- ============================================================================


-- 1. Panel snapshot ----------------------------------------------------------
-- Population overview. The FILTER clause is DuckDB/Postgres-style conditional
-- aggregation -- cleaner than CASE WHEN inside each aggregate.
SELECT count(DISTINCT PAT_ID)                                   AS patients,
       count(*)                                                 AS encounters,
       count(*) FILTER (WHERE SEX_NAME = 'Female')              AS female_enc,
       round(avg(PAT_AGE), 1)                                   AS avg_age,
       count(*) FILTER (WHERE PAT_AGE >= 65)                    AS medicare_age_enc,
       count(DISTINCT VISIT_PROV_ID)                            AS providers,
       min(CONTACT_DATE) || ' .. ' || max(CONTACT_DATE)         AS date_range
FROM v_encounter;


-- 2. Chronic condition registry ----------------------------------------------
-- Prevalence across the panel. v_diagnosis unions the encounter, problem-list
-- and medical-history sources, so DISTINCT PAT_ID is required.
SELECT dx_name,
       icd10,
       count(DISTINCT PAT_ID)                                   AS patients,
       round(100.0 * count(DISTINCT PAT_ID) / 100, 1)           AS pct_of_panel
FROM v_diagnosis
GROUP BY 1, 2
HAVING count(DISTINCT PAT_ID) >= 5
ORDER BY patients DESC
LIMIT 15;


-- 3. Uncontrolled hypertension -----------------------------------------------
-- Care gap: carries a hypertension diagnosis AND presented above goal.
SELECT e.PAT_NAME,
       e.PAT_AGE,
       v.systolic || '/' || v.diastolic                         AS bp,
       v.bmi,
       e.CONTACT_DATE,
       e.visit_prov_name
FROM v_encounter e
JOIN v_vitals v
  ON v.PAT_ID = e.PAT_ID AND v.RECORD_DATE = e.CONTACT_DATE
WHERE (v.systolic >= 140 OR v.diastolic >= 90)
  AND EXISTS (SELECT 1 FROM v_diagnosis d
              WHERE d.PAT_ID = e.PAT_ID AND d.dx_name ILIKE '%hypertens%')
ORDER BY v.systolic DESC
LIMIT 10;


-- 4. Heart failure without guideline-directed therapy ------------------------
-- A real quality measure: HFrEF patients should be on a beta-blocker plus a
-- renin-angiotensin agent. list_sort(list(DISTINCT ...)) collapses each
-- patient's drug classes into one sorted, deterministic array.
WITH hf AS (
    SELECT DISTINCT PAT_ID FROM v_diagnosis WHERE dx_name ILIKE '%heart failure%'
), rx AS (
    SELECT PAT_ID, list_sort(list(DISTINCT generic_class)) AS classes
    FROM v_medication GROUP BY 1
), assessed AS (
    SELECT p.PAT_NAME,
           p.PAT_AGE,
           coalesce(rx.classes, [])                                 AS med_classes,
           list_contains(coalesce(rx.classes, []), 'Beta-blocker')  AS on_betablocker,
           len(list_filter(coalesce(rx.classes, []),
               x -> x IN ('ACEi', 'ARB', 'ARNi'))) > 0              AS on_raas_agent
    FROM hf
    JOIN patient p USING (PAT_ID)
    LEFT JOIN rx USING (PAT_ID)
)
SELECT * FROM assessed
WHERE NOT (on_betablocker AND on_raas_agent)     -- show only the gaps
ORDER BY PAT_NAME;


-- 5. Atrial fibrillation without anticoagulation -----------------------------
-- Stroke-prevention gap. Same shape as query 4, different measure.
SELECT p.PAT_NAME, p.PAT_AGE,
       string_agg(DISTINCT d.dx_name, '; ')                     AS afib_dx
FROM v_diagnosis d
JOIN patient p USING (PAT_ID)
WHERE d.dx_name ILIKE '%atrial fibrillation%'
  AND NOT EXISTS (SELECT 1 FROM v_medication m
                  WHERE m.PAT_ID = d.PAT_ID
                    AND m.generic_class IN ('DOAC', 'VKA'))
GROUP BY 1, 2
ORDER BY p.PAT_AGE DESC;


-- 6. Lab trajectory: biggest movers ------------------------------------------
-- The one genuine change-over-time story in this dataset. first_value and
-- last_value over each patient-analyte series, then QUALIFY to keep one row
-- per series -- no subquery needed to filter on a window function.
SELECT p.PAT_NAME,
       r.COMPONENT_NAME                                         AS analyte,
       first_value(r.value) OVER w                              AS first_value,
       last_value(r.value)  OVER w                              AS latest_value,
       round(last_value(r.value) OVER w - first_value(r.value) OVER w, 2) AS delta,
       min(r.RESULT_DATE) OVER w || ' .. ' || max(r.RESULT_DATE) OVER w   AS span
FROM v_lab_result r
JOIN patient p USING (PAT_ID)
WHERE r.COMPONENT_NAME IN ('HbA1c', 'LDL Cholesterol', 'eGFR', 'NT-proBNP')
WINDOW w AS (PARTITION BY r.PAT_ID, r.COMPONENT_NAME ORDER BY r.RESULT_DATE
             ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
QUALIFY row_number() OVER (PARTITION BY r.PAT_ID, r.COMPONENT_NAME) = 1
    AND count(*)      OVER w > 1
ORDER BY abs(delta) DESC
LIMIT 12;


-- 7. Outstanding lab orders --------------------------------------------------
-- Operational view: ordered at the AWV, still unresulted. This is why
-- v_lab_order exists separately from v_lab_result.
SELECT o.ORD_PROV_NAME                                          AS provider,
       o.ORD_PROV_SPECIALTY                                     AS specialty,
       count(*)                                                 AS pending,
       string_agg(DISTINCT o.test_name, ', ')                   AS tests,
       min(o.ORDER_DATE)                                        AS oldest_order
FROM v_lab_order o
WHERE o.is_pending
GROUP BY 1, 2
ORDER BY pending DESC;


-- 8. Abnormal lab burden -----------------------------------------------------
-- Which analytes flag abnormal most often, and how far out of range they run.
SELECT COMPONENT_NAME                                           AS analyte,
       count(*)                                                 AS results,
       count(*) FILTER (WHERE is_abnormal)                      AS abnormal,
       round(100.0 * avg(is_abnormal::INT), 1)                  AS pct_abnormal,
       round(avg(value) FILTER (WHERE is_abnormal), 2)          AS avg_abnormal_value,
       any_value(REFERENCE_LOW) || '-' || any_value(REFERENCE_HIGH) AS ref_range,
       any_value(unit)                                          AS unit
FROM v_lab_result
GROUP BY 1
HAVING count(*) >= 10
ORDER BY pct_abnormal DESC
LIMIT 12;


-- 9. Polypharmacy ------------------------------------------------------------
-- Patients on 5+ concurrent medications, with their regimen inline.
SELECT p.PAT_NAME,
       p.PAT_AGE,
       count(DISTINCT m.MEDICATION_ID)                          AS med_count,
       round(sum(m.DAILY_FREQUENCY), 1)                         AS doses_per_day,
       string_agg(DISTINCT m.DISPLAY_NAME, ' | ' ORDER BY m.DISPLAY_NAME) AS regimen
FROM v_medication m
JOIN patient p USING (PAT_ID)
GROUP BY 1, 2
HAVING count(DISTINCT m.MEDICATION_ID) >= 5
ORDER BY med_count DESC
LIMIT 10;


-- 10. Provider scorecard -----------------------------------------------------
-- Panel size and lab throughput per provider. Joins the two lab views side by
-- side without pretending either is the other's parent.
SELECT e.visit_prov_name                                        AS provider,
       e.visit_prov_specialty                                   AS specialty,
       count(DISTINCT e.PAT_ENC_CSN_ID)                         AS visits,
       count(DISTINCT o.ORDER_PROC_ID)                          AS awv_labs_ordered,
       count(DISTINCT o.ORDER_PROC_ID) FILTER (WHERE o.is_pending) AS still_pending,
       round(100.0 * avg(r.is_abnormal::INT), 1)                AS pct_abnormal_hx_labs
FROM v_encounter e
LEFT JOIN v_lab_order  o ON o.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID
LEFT JOIN v_lab_result r ON r.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID
GROUP BY 1, 2
ORDER BY visits DESC
LIMIT 10;


-- 11. Note search ------------------------------------------------------------
-- Free-text progress notes. Useful for showing unstructured data alongside the
-- structured tables.
SELECT p.PAT_NAME,
       n.ENTRY_TIME::DATE                                       AS noted,
       n.AUTHOR_PROV_NAME                                       AS author,
       regexp_extract(n.NOTE_TEXT, 'BP: ([0-9]+/[0-9]+)', 1)    AS bp_in_note,
       substr(n.NOTE_TEXT, 1, 90) || '...'                      AS excerpt
FROM hno_info n
JOIN patient p USING (PAT_ID)
WHERE n.NOTE_TEXT ILIKE '%asthma%'
ORDER BY noted DESC
LIMIT 5;


-- 12. Note extraction audit --------------------------------------------------
-- The progress notes are templated prose generated from the structured tables,
-- so v_note_extract can be scored against a known answer key. Every field comes
-- back at 100% -- which is the point: it demonstrates extraction mechanics
-- against ground truth, not NLP finding something the coded data missed.
WITH dx AS (
    SELECT PAT_ENC_CSN_ID, list_sort(list(DISTINCT DX_NAME))      AS coded
    FROM pat_enc_dx GROUP BY 1
), rx AS (
    SELECT PAT_ENC_CSN_ID, list_sort(list(DISTINCT DISPLAY_NAME)) AS coded
    FROM order_med GROUP BY 1
), scored AS (
    SELECT n.systolic  = v.systolic  AND n.diastolic = v.diastolic AS bp_ok,
           abs(n.bmi - v.bmi) < 0.05                               AS bmi_ok,
           list_sort(n.conditions)  = dx.coded                     AS dx_ok,
           list_sort(n.medications) = rx.coded                     AS rx_ok
    FROM v_note_extract n
    JOIN v_encounter e USING (PAT_ENC_CSN_ID)
    JOIN v_vitals    v ON v.PAT_ID = n.PAT_ID AND v.RECORD_DATE = e.CONTACT_DATE
    LEFT JOIN dx USING (PAT_ENC_CSN_ID)
    LEFT JOIN rx USING (PAT_ENC_CSN_ID)
)
SELECT field, notes, matched, round(100.0 * matched / notes, 1) AS pct_agreement
FROM (
    SELECT 'blood pressure' AS field, count(*) AS notes, count(*) FILTER (WHERE bp_ok)  AS matched FROM scored
    UNION ALL SELECT 'BMI',            count(*), count(*) FILTER (WHERE bmi_ok) FROM scored
    UNION ALL SELECT 'conditions',     count(*), count(*) FILTER (WHERE dx_ok)  FROM scored
    UNION ALL SELECT 'medications',    count(*), count(*) FILTER (WHERE rx_ok)  FROM scored
)
ORDER BY field;
