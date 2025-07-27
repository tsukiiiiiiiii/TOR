from agents_2 import BaseDoctor, ChiefComplaintDoctor, LabDoctor, ImagingDoctor, PathologyDoctor
import datetime
import json
import random
from prettytable import PrettyTable
from termcolor import cprint
import os
import glob
from multiprocessing import Pool
from openai import OpenAI

import traceback  
from functools import partial 
import time

MODEL_NAME = "deepseek-chat"  

ERROR_LOG = "error_log.txt"
MAX_RETRIES = 3

# error logging
def log_error(error_msg, case_file):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] File: {case_file}\n")
        f.write(f"Error: {error_msg}\n\n")

def safe_process_case(case_file):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            result = process_case(case_file)
            return result
        except Exception as e:
            retries += 1
            error_msg = f"Attempt {retries} failed: {str(e)}\n{traceback.format_exc()}"
            print(f"Error processing {case_file}: {error_msg}")
            
            if retries == MAX_RETRIES:
                log_error(error_msg, case_file)
                print(f"Max retries reached for {case_file}, logging error")
                return None
            else:
                print(f"Retrying {case_file} ({retries}/{MAX_RETRIES})...")
                time.sleep(2 ** retries) 
    return None


def chat(cont):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": cont},
                {"role": "user", "content": "Please carefully consider and answer the above questions."}
            ],
            stream=False
        )

        result = response.choices[0].message.content

        return result
    except Exception as e:
        print(f"API Error calling or parsing response: {str(e)}")
        return None

class MedicalTeam:
    def __init__(self, patient_case):
        self.patient_case = patient_case
        self.doctors = {
            "chief_complaint": ChiefComplaintDoctor(),
            "lab": LabDoctor(),
            "imaging": ImagingDoctor(),
            "pathology": PathologyDoctor()
        }
        self.agent_emoji = [
            '\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F',
            '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F'
        ]
        random.shuffle(self.agent_emoji)
        
        self.interaction_log = {}
        self.round_opinions = {}
        self.options = patient_case[1].get("options", "No specific options available")

    # Obtain initial diagnoses from various doctors 
    def get_initial_diagnoses(self, original_filename):
        print("Obtaining initial diagnosis~")
        base_filename = os.path.basename(original_filename)
        filename = f"step2_{base_filename}"
        result_dir = "result"
        filepath = os.path.join(result_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        diagnoses = {}
        retrieved_info = {}
        if self.patient_case[0]["patient_info"]:
            ans = self.doctors["chief_complaint"].examine_patient(
                self.patient_case[0]["patient_info"]
            )
            diagnoses["chief_complaint"] = ans["response"]
            retrieved_info["chief_complaint"] = ans["retrieved_info"]
            
        if self.patient_case[0]["lab_results"]:
            ans = self.doctors["lab"].analyze_results(
                self.patient_case[0]["lab_results"]
            )
            diagnoses["lab"] = ans["response"]
            retrieved_info["lab"] = ans["retrieved_info"]
                
        if self.patient_case[0]["imaging_results"]:
            ans = self.doctors["imaging"].analyze_images(
                self.patient_case[0]["imaging_results"]
            )
            diagnoses["imaging"] = ans["response"]
            retrieved_info["imaging"] = ans["retrieved_info"]
            
        if self.patient_case[0]["pathology_results"]:
            ans = self.doctors["pathology"].analyze_pathology(
                self.patient_case[0]["pathology_results"]
            )
            diagnoses["pathology"] = ans["response"]
            retrieved_info["pathology"] = ans["retrieved_info"]
            
        os.makedirs(result_dir, exist_ok=True) 
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(diagnoses, f, ensure_ascii=False, indent=2)

        retrieved_info_filename = f"retrieved_info_{base_filename}"
        retrieved_info_filepath = os.path.join(result_dir, retrieved_info_filename)
        
        with open(retrieved_info_filepath, 'w', encoding='utf-8') as f:
            json.dump(retrieved_info, f, ensure_ascii=False, indent=2)
        
        print(diagnoses)
        return diagnoses

    # Conduct multiple rounds of discussion
    def conduct_discussion(self, case_file, num_rounds=1, num_turns=1):
        print("\n=== Start doctor team discussion ===")
        
        initial_diagnoses = self.get_initial_diagnoses(case_file)
        active_doctors = {k: v for k, v in initial_diagnoses.items() if v is not None}
        
        self.interaction_log = {
            f'Round {round_num}': {
                f'Turn {turn_num}': {
                    source_doctor: {target_doctor: None 
                        for target_doctor in active_doctors.keys()
                    } for source_doctor in active_doctors.keys()
                } for turn_num in range(1, num_turns + 1)
            } for round_num in range(1, num_rounds + 1)
        }
        
        self.round_opinions = {round_num: {} for round_num in range(1, num_rounds + 1)}
        self.round_opinions[1] = initial_diagnoses
        
        # Conduct multiple rounds of discussion
        for round_num in range(1, num_rounds + 1):
            print(f"\n== Round {round_num} Discussion ==")
            
            for turn_num in range(1, num_turns + 1):
                print(f"\n- Round {turn_num} -")
                
                for source_doctor, source_opinion in active_doctors.items():
                    source_emoji = self.agent_emoji[list(active_doctors.keys()).index(source_doctor)]
                    
                    discussion_prompt = self._generate_discussion_prompt(
                        round_num, turn_num, source_doctor, active_doctors
                    )
                    
                    participate = self._should_participate(source_doctor, discussion_prompt)
                    
                    if participate:
                        target_doctors = self._choose_discussion_targets(
                            source_doctor, active_doctors.keys()
                        )
                        
                        for target_doctor in target_doctors:
                            target_emoji = self.agent_emoji[list(active_doctors.keys()).index(target_doctor)]
                            
                            opinion = self._generate_opinion(
                                source_doctor, target_doctor, discussion_prompt
                            )
                            
                            self.interaction_log[f'Round {round_num}'][f'Turn {turn_num}'][source_doctor][target_doctor] = opinion
                            
                            print(f" {source_emoji} {source_doctor} -> {target_emoji} {target_doctor}: {opinion}")
                    else:
                        print(f" {source_emoji} {source_doctor}: \U0001f910 (Not participate in the discussion this round.)")
            
            updated_opinions = self._collect_updated_opinions(round_num, active_doctors)
            self.round_opinions[round_num + 1] = updated_opinions
            
        return self._make_final_decision()

    # Generate discussion prompts
    def _generate_discussion_prompt(self, round_num, turn_num, source_doctor, active_doctors):
        prompt = f"""
1. Patient case:
{json.dumps(self.patient_case[0], ensure_ascii=False, indent=2)}

2. Current round: Round {round_num}, Turn {turn_num}

3. Diagnosis opinions from each doctor:
"""
        for doctor, opinion in self.round_opinions[round_num].items():
            prompt += f"\n{doctor}: {opinion}"
                
        return prompt    

    # Deciding whether to participate in the discussion
    def _should_participate(self, doctor_type, prompt):
        participation_prompt = f"""
Based on the current discussion situation, actively identify areas where your perspective differs from others'. Consider if providing your unique viewpoint could help resolve disagreements or improve the diagnosis. You should participate whenever there's an opportunity to clarify your position or persuade others, even if some opinions have already been expressed.
Do you need to provide new insights or engage in discussion with other doctors?
Please answer only with "Yes" or "No".

Current situation:
{prompt}
"""
        response = chat(participation_prompt)
        return "Yes" in response

    # Choose which doctors to discuss with
    def _choose_discussion_targets(self, source_doctor, available_doctors):
        targets = []
        for doctor in available_doctors:
            if doctor != source_doctor:
                if random.random() < 0.7:  # 70% chance to discuss with other doctors
                    targets.append(doctor)
        return targets

    # Generate opinions for specific target doctors
    def _generate_opinion(self, source_doctor, target_doctor, prompt):
        
        opinion_prompt = f"""As the {source_doctor} doctor, please provide your professional opinion on the diagnosis from the {target_doctor} doctor:

{prompt}

Please concisely express your views, focusing on:
1. Which aspects of the other doctor's opinion you agree or disagree with
2. What additional insights or suggestions you have based on your expertise
3. How to integrate both professional perspectives to improve the diagnosis
"""
        return chat(opinion_prompt)

    # Collect and update the viewpoints after this round of discussion
    def _collect_updated_opinions(self, round_num, active_doctors):
        updated_opinions = {}
        
        for doctor_type in active_doctors.keys():
            update_prompt = f"""As the {doctor_type} doctor, please generate an UPDATED diagnostic tree based on original assessment and new feedback. 

Output Format:       
{doctor_type} Doctor Reasoning Pathway
├── Disease 1
│   └── Analysis: ...
│       ├── Evidence 1: ...
│       ├── Evidence 2: ...
│       └── Evidence 3: ...
├── Disease 2
│   └── Analysis: ...
│       ├── Evidence 1: ...
│       ├── Evidence 2: ...
│       └── Evidence 3: ...
└── ...
Please ensure that the output strictly follows the above format and only includes the evidence tree structure. Avoid any additional text or explanations outside the tree structure.

1. Original diagnosis:
{self.round_opinions[1][doctor_type]}

2. Feedback received in this round:
"""

            for turn in self.interaction_log[f'Round {round_num}'].values():
                for source, targets in turn.items():
                    if targets[doctor_type]:
                        update_prompt += f"\n{source}: {targets[doctor_type]}"
                        
            updated_opinion = chat(update_prompt)
            updated_opinions[doctor_type] = updated_opinion
            
        return updated_opinions

    # Make the final decision
    def _make_final_decision(self):
        final_prompt = f"""As the head of the medical team, please make the final diagnosis based on the following information:

1. Patient case:
{json.dumps(self.patient_case[0], ensure_ascii=False, indent=2)}

2. Diagnosis opinions from the last round:
"""
        last_round_opinions = self.round_opinions[max(self.round_opinions.keys())]
        for doctor, opinion in last_round_opinions.items():
            final_prompt += f"{doctor}: {opinion}\n"
            
        final_prompt += f"\n3. Diagnosis options:\n{self.options}"

        final_prompt += """
The output should include:
1. The final diagnosis result (please select the appropriate letter from the options)
2. The evidence tree structure, formatted as follows:
Reasoning Pathway
├── Disease 1
│   └── Analysis: ...
│       ├── Evidence 1: ...
│       ├── Evidence 2: ...
│       └── Evidence 3: ...
├── Disease 2
│   └── Analysis: ...
│       ├── Evidence 1: ...
│       ├── Evidence 2: ...
│       └── Evidence 3: ...
└── ...
The result should be output in JSON format, strictly following the format below. Do not add any extraneous words!
{
    "selected_options":"",
    "evi_tree":""
}
"""
        
        final_decision = chat(final_prompt)
        return final_decision

    # Visualize the interaction between doctors
    def visualize_interactions(self):
        active_doctors = list(self.doctors.keys())
        table = PrettyTable([''] + [f"{doc} ({self.agent_emoji[i]})" for i, doc in enumerate(active_doctors)])
        
        for i, source in enumerate(active_doctors):
            row = [f"{source} ({self.agent_emoji[i]})"]
            for j, target in enumerate(active_doctors):
                if source == target:
                    row.append(' ')
                else:
                    source_to_target = False
                    target_to_source = False
                    
                    for round_data in self.interaction_log.values():
                        for turn_data in round_data.values():
                            if turn_data.get(source, {}).get(target):
                                source_to_target = True
                            if turn_data.get(target, {}).get(source):
                                target_to_source = True
                    
                    if not source_to_target and not target_to_source:
                        row.append(' ')
                    elif source_to_target and not target_to_source:
                        row.append(f'\u270B ({i+1}->{j+1})')
                    elif target_to_source and not source_to_target:
                        row.append(f'\u270B ({i+1}<-{j+1})')
                    else:
                        row.append(f'\u270B ({i+1}<->{j+1})')
            
            table.add_row(row)
        
        print("\n=== Interaction among doctor teams ===")
        print(table)

def process_case(case_file):
    print("Reading case files~")
    with open(case_file, 'r', encoding='utf-8') as f:
        case_data = json.load(f)
    
    processed_case = [
        {
            "patient_info": {
                "Age": case_data.get("Age", ""),
                "Sex": case_data.get("Sex", ""),
                "Chief-Complaints": case_data.get("Chief-Complaints", ""),
                "Present-Illness": case_data.get("Present-Illness", ""),
                "Physical-Examination": case_data.get("Physical-Examination", "")
            },
            "lab_results": case_data.get("Laboratory-Examination", ""),
            "imaging_results": "\n".join([
                case_data.get("X光影像检查", ""),
                case_data.get("CT影像检查", ""),
                case_data.get("磁共振影像检查", ""),
                case_data.get("超声影像检查", "")
            ]).strip(),
            "pathology_results": case_data.get("病理检查", "")
        },{
            "diagnosis": case_data.get("Diagnosis", ""),
            "options": case_data.get("options", ""),
            "label": case_data.get("label", "")
        }
    ]
    base_filename = os.path.basename(case_file)
    filename = f"step3_{base_filename}"
    result_dir = "result"
    filepath = os.path.join(result_dir, filename)
    
    if os.path.exists(filepath):
        print(f"The result file already exists, read directly: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return result
    
    try:
        print("Creating medical team and discussing~")

        team = MedicalTeam(processed_case)
        final_decision = team.conduct_discussion(case_file)
    except Exception as e:
        print(f"An uncaught exception occurred during case handling: {str(e)}")
        raise 

    team.visualize_interactions()
    
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "case_info": processed_case,
        "discussion_process": {
            "round_opinions": team.round_opinions
        },
        "final_decision": final_decision
    }
    
    os.makedirs(result_dir, exist_ok=True) 
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nThe result has been saved to: {filename}")
    return result



if __name__ == "__main__":  
    client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write("Error Log\n---------\n\n")

    case_dir = "YOUR_INPUT_DATA_PATH"
    case_files = glob.glob(os.path.join(case_dir, "*.json"))
    print("Processing~")

    with Pool(processes=2) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(safe_process_case, case_files)):
            if result is not None:
                results.append(result)
            print(f"Completed {i+1}/{len(case_files)} files")

    print(f"Processing completed, see error log: {ERROR_LOG}")