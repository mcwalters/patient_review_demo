"""Rules every agent in this prototype has to follow.

These lived inline in each agent's instruction, which is how they drifted: the
naming rule was copied verbatim three times, SEVERITY_WORDS reached two
specialists of three, and the pre-visit brief -- written last -- reintroduced
the honorific bug that had already been fixed in the panel agents. A rule that
exists in four places is four rules.

Every instruction now composes from here. Adding a rule means adding it once.
"""
from __future__ import annotations

NAMING = """\
NAME PEOPLE AS THE RECORD DOES. Use the patient's name exactly as stored and add
nothing to it -- no Mr, Ms, Mrs or Dr. There is a sex field if it ever matters;
inferring one from a first name is how a clinical tool misgenders somebody. A
finding without names cannot be acted on, so never describe a group of patients
without listing them.
"""

NUMBERS = """\
EVERY NUMBER YOU WRITE MUST COME FROM A TOOL RESULT OR THE DATA YOU WERE GIVEN.
Not a count, not a lab value, not a duration, not a date. Quote them exactly. If
you want a number you were not given, leave it out or call the tool that has it.
A run once reported "nineteen patients with impossible sodium" where the store
held thirteen, and the sentence around it was otherwise correct -- which is what
makes this failure hard to catch by reading.
"""

NO_STOP_DATES = """\
A PATIENT ON TWO AGENTS OF ONE CLASS IS NOT EVIDENCE OF CONCURRENT THERAPY.
Nothing in this extract is ever recorded as stopped -- END_DATE and DISCON_TIME
are empty and all 522 orders read Active -- so a switch and a combination look
identical. Across the 20 duplicated patient-class pairs the start dates are 113
to 1376 days apart, mean 814; Rogers, Jessica's four statin orders span 2022 to
2026. Call medication_timeline before describing anything as duplicate, double
or triple therapy, and report what the dates show. The reportable defect is the
missing discontinuation data, not the patient.
"""

BP_SEVERITY = """\
DO NOT INVENT SEVERITY LABELS FOR BLOOD PRESSURE. Call blood_pressure_staging
and use the stage it returns. A run reported "unaddressed severe hypertension"
for five patients and ranked them first through fifth; one met a severe
threshold and one was 111/104, a pulse pressure of 7, which is not a blood
pressure at all. Only 5 of 100 patients reach hypertensive crisis. The same tool
flags readings whose pulse pressure is impossible -- those are data defects and
belong in a data-quality finding, not a clinical one.
"""

# Everything an agent in this prototype must follow, regardless of its job.
SHARED = "\n" + NAMING + "\n" + NUMBERS + "\n" + NO_STOP_DATES + "\n" + BP_SEVERITY
