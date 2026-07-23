"""
Hand-curated evaluation test set for MedGuard RAG.

20 queries chosen to test specific system behaviors:
  - Structured drug-pair lookups (DDInter severity)
  - Single-drug label questions (pregnancy, kidneys, warnings)
  - Genuinely ambiguous queries (correct answer: refuse)
  - Out-of-scope queries (drug not in DB)
  - Edge cases (multiple drugs mentioned, brand vs. generic names)

Each entry has:
  question: the user-facing query
  ground_truth: a short reference answer built from the same FDA labels
                the RAG system has access to -- so we're measuring
                "does the system extract the right information from its
                own corpus," not "does it match some outside oracle."
  reference_drugs: which drug(s) SHOULD be identified (used for retrieval
                   diagnostics, not for scoring the answer itself)
  category: coarse test bucket, for reporting metrics by category

Kept as plain Python so the test set is diffable in git and reviewable
without opening a JSON editor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TestCase:
    question: str
    ground_truth: str
    reference_drugs: list[str]
    category: str  # "structured_pair" | "single_drug" | "refusal" | "out_of_scope"


TEST_SET: list[TestCase] = [
    # --- Structured drug-pair interactions (should surface DDInter severity + FDA label reasoning)
    TestCase(
        question="Does Warfarin interact with Ibuprofen?",
        ground_truth=(
            "Yes, this is a Major severity interaction per DDInter. NSAIDs including "
            "ibuprofen increase bleeding risk when combined with warfarin through "
            "platelet inhibition and other mechanisms. Close INR monitoring and "
            "watching for signs of bleeding are recommended."
        ),
        reference_drugs=["Warfarin", "Ibuprofen"],
        category="structured_pair",
    ),
    TestCase(
        question="Is it safe to take Warfarin with Naproxen?",
        ground_truth=(
            "This is a Major interaction. NSAIDs like naproxen have a synergistic "
            "effect with warfarin on bleeding risk. Concomitant use requires close "
            "monitoring."
        ),
        reference_drugs=["Warfarin", "Naproxen"],
        category="structured_pair",
    ),
    TestCase(
        question="Can I combine Warfarin and Celecoxib?",
        ground_truth=(
            "Celecoxib and anticoagulants including warfarin have a synergistic "
            "effect on bleeding per FDA labeling. This is a documented interaction "
            "requiring careful monitoring."
        ),
        reference_drugs=["Warfarin", "Celecoxib"],
        category="structured_pair",
    ),

    # --- Single-drug questions on drugs known to be in our corpus
    TestCase(
        question="Is Warfarin safe during pregnancy?",
        ground_truth=(
            "Warfarin is contraindicated during pregnancy except in pregnant women "
            "with mechanical heart valves at high risk of thromboembolism. Warfarin "
            "exposure during pregnancy causes a recognized pattern of major "
            "congenital malformations (warfarin embryopathy and fetotoxicity)."
        ),
        reference_drugs=["Warfarin"],
        category="single_drug",
    ),
    TestCase(
        question="What are the kidney warnings for Indomethacin?",
        ground_truth=(
            "Indomethacin can cause renal toxicity, particularly in patients where "
            "renal prostaglandins have a compensatory role in maintaining renal "
            "perfusion (e.g. elderly, volume-depleted, or those on diuretics). "
            "Renal function should be monitored in patients with renal impairment."
        ),
        reference_drugs=["Indomethacin"],
        category="single_drug",
    ),
    TestCase(
        question="What is the dosing for Mannitol?",
        ground_truth=(
            "Mannitol dosing should be individualized based on patient response and "
            "fluid status. Adult IV doses typically range from 50 to 200 g over "
            "24 hours."
        ),
        reference_drugs=["Mannitol"],
        category="single_drug",
    ),
    TestCase(
        question="What are the contraindications for Warfarin?",
        ground_truth=(
            "Warfarin is contraindicated in pregnancy (except for women with "
            "mechanical heart valves at high thromboembolic risk), in patients with "
            "hemorrhagic tendencies or blood dyscrasias, and in situations where "
            "the risk of bleeding outweighs the benefit of anticoagulation."
        ),
        reference_drugs=["Warfarin"],
        category="single_drug",
    ),
    TestCase(
        question="Does Ketorolac cause bleeding?",
        ground_truth=(
            "Yes, ketorolac (an NSAID) increases bleeding risk, particularly when "
            "combined with anticoagulants or other drugs affecting hemostasis. It "
            "carries the standard NSAID class warnings around GI and other bleeding."
        ),
        reference_drugs=["Ketorolac"],
        category="single_drug",
    ),
    TestCase(
        question="What are the serious warnings for Celecoxib?",
        ground_truth=(
            "Celecoxib carries warnings for hepatotoxicity (liver injury), "
            "cardiovascular thrombotic events including MI and stroke, GI bleeding "
            "and ulceration, and renal effects. Standard NSAID class boxed warning "
            "on CV and GI risk applies."
        ),
        reference_drugs=["Celecoxib"],
        category="single_drug",
    ),
    TestCase(
        question="Can Diclofenac cause liver problems?",
        ground_truth=(
            "Yes, diclofenac can cause hepatotoxicity. Patients should be informed "
            "of warning signs and symptoms of liver injury, and treatment should be "
            "discontinued if abnormal liver tests persist or worsen."
        ),
        reference_drugs=["Diclofenac"],
        category="single_drug",
    ),

    # --- Correct-refusal cases (evidence is thin or absent -- system should decline)
    TestCase(
        question="Is it safe to take a drug I did not name?",
        ground_truth=(
            "The system should decline to answer because no specific drug has been "
            "named. It should explain that a drug name is required and no "
            "meaningful safety assessment can be made without one."
        ),
        reference_drugs=[],
        category="refusal",
    ),
    TestCase(
        question="What should I do?",
        ground_truth=(
            "The system should decline because the question lacks any specific "
            "clinical context (no drug, condition, or symptom mentioned). It should "
            "ask for more specific information."
        ),
        reference_drugs=[],
        category="refusal",
    ),
    TestCase(
        question="Tell me everything about medicine.",
        ground_truth=(
            "The system should decline because the question is too broad and lacks "
            "specific drugs, conditions, or clinical context to ground a useful "
            "answer in the retrieved evidence."
        ),
        reference_drugs=[],
        category="refusal",
    ),

    # --- Out-of-scope: drugs not in our corpus at all
    TestCase(
        question="What are the side effects of a made-up drug called Zorbatrix?",
        ground_truth=(
            "The system should indicate that Zorbatrix is not in its available "
            "sources and it cannot provide information about a drug it has no "
            "evidence for."
        ),
        reference_drugs=[],
        category="out_of_scope",
    ),
    TestCase(
        question="Does this drug interact with kryptonite?",
        ground_truth=(
            "The system should decline: kryptonite is not a real drug or substance "
            "in the FDA corpus, and no drug is named in the query either."
        ),
        reference_drugs=[],
        category="out_of_scope",
    ),

    # --- Edge cases
    TestCase(
        question="What are the dosage guidelines for Ibuprofen in elderly patients?",
        ground_truth=(
            "Elderly patients are at increased risk of NSAID adverse effects "
            "including GI bleeding and renal impairment. Ibuprofen should be used "
            "at the lowest effective dose for the shortest duration, with careful "
            "monitoring."
        ),
        reference_drugs=["Ibuprofen"],
        category="single_drug",
    ),
    TestCase(
        question="Can pregnant women take Diclofenac?",
        ground_truth=(
            "Diclofenac use should generally be avoided in pregnancy, particularly "
            "after 20 weeks gestation due to risk of fetal renal dysfunction and "
            "oligohydramnios, and after 30 weeks due to risk of premature closure "
            "of the ductus arteriosus."
        ),
        reference_drugs=["Diclofenac"],
        category="single_drug",
    ),
    TestCase(
        question="Are there any drug interactions I should know about with Rosuvastatin?",
        ground_truth=(
            "Rosuvastatin has documented interactions with cyclosporine, gemfibrozil, "
            "protease inhibitors, and warfarin (which can increase INR). Combining "
            "with other statins or fibrates increases risk of myopathy and "
            "rhabdomyolysis."
        ),
        reference_drugs=["Rosuvastatin"],
        category="single_drug",
    ),
    TestCase(
        question="What are the warnings for Phenytoin?",
        ground_truth=(
            "Phenytoin carries serious warnings including risk of severe skin "
            "reactions (Stevens-Johnson syndrome, toxic epidermal necrolysis), "
            "hepatotoxicity, and hematopoietic complications. Serum concentrations "
            "should be monitored due to its narrow therapeutic index."
        ),
        reference_drugs=["Phenytoin"],
        category="single_drug",
    ),
    TestCase(
        question="Does Sertraline cause serotonin syndrome?",
        ground_truth=(
            "Yes, sertraline (an SSRI) can cause potentially life-threatening "
            "serotonin syndrome, particularly when combined with other serotonergic "
            "drugs such as SNRIs, triptans, MAOIs, or tramadol."
        ),
        reference_drugs=["Sertraline"],
        category="single_drug",
    ),
]
