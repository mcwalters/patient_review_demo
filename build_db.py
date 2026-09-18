"""Build a self-contained DuckDB database from the EHR CSVs in ./data.

    pip install duckdb && python build_db.py

Produces ehr.duckdb: 14 raw tables + analyst views. Re-runnable (drops and rebuilds).
"""
import duckdb, glob, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
DB = os.path.join(ROOT, "ehr.duckdb")

# Columns that are 100% empty in the source and so land as VARCHAR unless forced.
FORCE_TYPES = {
    "patient":      {"DEATH_DATE": "DATE"},
    "order_med":    {"END_DATE": "DATE", "DISCON_TIME": "TIMESTAMP", "MED_COMMENTS": "VARCHAR"},
    "medical_hx":   {"MED_HX_END_DT": "DATE", "COMMENTS": "VARCHAR", "MED_HX_ANNOTATION": "VARCHAR"},
    "surgical_hx":  {"SURG_HX_END_DT": "DATE", "COMMENTS": "VARCHAR"},
    "problem_list": {"NOTED_END_DATE": "DATE", "PROBLEM_CMT": "VARCHAR"},
    "ip_flwsht_meas": {"MEAS_COMMENT": "VARCHAR"},
}

def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = duckdb.connect(DB)
    t0 = time.time()

    csvs = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    if not csvs:
        sys.exit(f"no CSVs found in {DATA}")

    for path in csvs:
        table = os.path.basename(path)[:-4]
        forced = FORCE_TYPES.get(table, {})
        types = ", ".join(f"'{c}': '{t}'" for c, t in forced.items())
        types_arg = f", types={{{types}}}" if types else ""
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_csv('{path}', header=true, sample_size=-1{types_arg})"
        )

    # --- analyst views -------------------------------------------------
    # Vitals are stored long-and-thin; BP is a non-numeric "sys/dia" string.
    con.execute("""
    CREATE VIEW v_vitals AS
    SELECT PAT_ID, PAT_MRN_ID, RECORD_DATE,
           max(CASE WHEN FLO_MEAS_NAME = 'BLOOD PRESSURE'
                    THEN CAST(split_part(MEAS_VALUE, '/', 1) AS INTEGER) END) AS systolic,
           max(CASE WHEN FLO_MEAS_NAME = 'BLOOD PRESSURE'
                    THEN CAST(split_part(MEAS_VALUE, '/', 2) AS INTEGER) END) AS diastolic,
           max(CASE WHEN FLO_MEAS_NAME = 'HEART RATE'       THEN MEAS_VALUE_NUM END) AS heart_rate,
           max(CASE WHEN FLO_MEAS_NAME = 'TEMPERATURE'      THEN MEAS_VALUE_NUM END) AS temperature,
           max(CASE WHEN FLO_MEAS_NAME = 'RESPIRATORY RATE' THEN MEAS_VALUE_NUM END) AS resp_rate,
           max(CASE WHEN FLO_MEAS_NAME = 'WEIGHT/SCALE'     THEN MEAS_VALUE_NUM END) AS weight_lb,
           max(CASE WHEN FLO_MEAS_NAME = 'SOM AMB R BMI'    THEN MEAS_VALUE_NUM END) AS bmi
    FROM ip_flwsht_meas
    GROUP BY 1, 2, 3
    """)

    # One row per encounter with patient + provider context.
    con.execute("""
    CREATE VIEW v_encounter AS
    SELECT e.PAT_ENC_CSN_ID, e.PAT_ID, e.PAT_MRN_ID, e.CONTACT_DATE, e.APPT_TIME,
           e.DEPARTMENT_NAME, e.LOC_NAME, e.INSURANCE,
           p.PAT_NAME, p.BIRTH_DATE, p.PAT_AGE, p.SEX_NAME,
           e.VISIT_PROV_ID, pr.PROV_NAME AS visit_prov_name, pr.SPECIALTY AS visit_prov_specialty,
           dx.DX_ICD_CODE AS primary_icd10, dx.DX_NAME AS primary_dx
    FROM pat_enc e
    JOIN patient p USING (PAT_ID)
    LEFT JOIN provider pr ON pr.PROV_ID = e.VISIT_PROV_ID
    LEFT JOIN pat_enc_dx dx ON dx.PAT_ENC_CSN_ID = e.PAT_ENC_CSN_ID AND dx.DX_SEQ = 1
    """)

    # Labs. NOTE: order_results does NOT join to order_proc_awv on ORDER_PROC_ID
    # (0 of 1288 match). Both tables carry the same LOINC universe, so this view
    # keys off order_results and links back to the encounter, not the order.
    con.execute("""
    CREATE VIEW v_lab_result AS
    SELECT r.RESULT_ID, r.PAT_ID, r.PAT_ENC_CSN_ID, r.COMPONENT_NAME, r.LOINC_CODE,
           r.ORD_NUM_VALUE AS value, r.REFERENCE_UNIT AS unit,
           r.REFERENCE_LOW, r.REFERENCE_HIGH,
           r.RESULT_FLAG_C, (r.RESULT_FLAG_C = 2) AS is_abnormal,
           r.RESULT_DATE, e.CONTACT_DATE AS encounter_date
    FROM order_results r
    LEFT JOIN pat_enc e ON e.PAT_ENC_CSN_ID = r.PAT_ENC_CSN_ID
    """)

    # Active medications with RxNorm.
    con.execute("""
    CREATE VIEW v_medication AS
    SELECT m.ORDER_MED_ID, m.PAT_ID, m.PAT_ENC_CSN_ID, m.MEDICATION_ID,
           m.DISPLAY_NAME, m.SIMPLE_GENERIC_C_NAME AS generic_class,
           m.HV_DISCRETE_DOSE AS dose, m.HV_DOSE_UNIT_C_NAME AS dose_unit,
           m.MED_ROUTE_C_NAME AS route, m.HV_DISCR_FREQ_ID_FREQ_NAME AS frequency,
           f.DAILY_FREQUENCY, x.RXNORM_CODE, m.START_DATE, m.ORDER_STATUS_C_NAME
    FROM order_med m
    LEFT JOIN rxnorm_codes x ON x.MEDICATION_ID = m.MEDICATION_ID
    LEFT JOIN ip_frequency f ON f.FREQ_NAME = m.HV_DISCR_FREQ_ID_FREQ_NAME
    """)

    # Diagnoses unioned across the three sources. DX_ID is row-random and does
    # NOT join across tables -- ICD10 is the only stable diagnosis key.
    con.execute("""
    CREATE VIEW v_diagnosis AS
    SELECT e.PAT_ID, d.PAT_ENC_CSN_ID, d.DX_ICD_CODE AS icd10, d.DX_NAME AS dx_name,
           d.DX_DATE AS dx_date, 'encounter' AS source
    FROM pat_enc_dx d JOIN pat_enc e USING (PAT_ENC_CSN_ID)
    UNION ALL
    SELECT PAT_ID, NULL, ICD10_CODE, PROBLEM_NAME, NOTED_DATE, 'problem_list'
    FROM problem_list
    UNION ALL
    SELECT PAT_ID, PAT_ENC_CSN_ID, ICD10_CODE, DX_NAME, MED_HX_START_DT, 'medical_hx'
    FROM medical_hx
    """)

    n = con.execute("SELECT count(*) FROM duckdb_tables() WHERE schema_name='main'").fetchone()[0]
    v = con.execute("SELECT count(*) FROM duckdb_views() WHERE schema_name='main' AND NOT internal").fetchone()[0]
    elapsed = time.time() - t0
    con.close()  # checkpoints; file size is only accurate afterwards
    print(f"built {DB}")
    print(f"  {n} tables, {v} views in {elapsed:.2f}s, {os.path.getsize(DB) / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
