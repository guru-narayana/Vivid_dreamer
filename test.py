# read the names of all folders in a directory and write them to a file

import os
import pandas as pd
import re
from PIL import Image
import ImageReward as RM
model = RM.load("ImageReward-v1.0")

def get_file_names(folder):
    files = os.listdir(folder)
    List = []
    for file in files:
        file = file.split('@')[0]
        prompt = file.replace('_', ' ')
        List.append((file, prompt))
    return List

def retrieve_image(base_directory, regex):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            full_path = os.path.join(root, file)
            if regex.match(full_path):
                img = Image.open(full_path)                 
                return img
    return None

def find_file_with_pattern(base = "sd",use_view_based_pormpting = False):
    base_directory = 'outputs/gaussiandreamer-' + base
    prompt_list = get_file_names(base_directory)
    df  = pd.DataFrame(columns=["Prompt", "Front view", "Side view 1", "Back view", "Side view 2"])
    for prompt_info in prompt_list:
        prompt_file, prompt = prompt_info
        image_ids = [0,30,60,90]
        prompt_extensions = [", front view", ", side view", ", back view", ", side view"]
        scores = []
        for index, image_id in enumerate(image_ids):
            if use_view_based_pormpting:
                prompt_view  = prompt + prompt_extensions[index]
            else:
                prompt_view = prompt
            pattern = r'.*/' +prompt_file + '@\d{8}-\d{6}/save/it\d+-test/'+ str(image_id) +'+\.png$'
            regex = re.compile(pattern)
            img = retrieve_image(base_directory, regex)
            if img:
                score = model.score(prompt_view, img)* 20 + 50
                scores.append(score)
            else:
                print("Invalid image path encountered : ",prompt_file,image_id)
        data = pd.DataFrame({'Prompt': [prompt], 'Front view': [scores[0]], 'Side view 1': [scores[1]],  "Back view" : [scores[2]], "Side view 2" : [scores[3]]})
        df = pd.concat([df,data], ignore_index=True)
    df = df.sort_values(df.columns[0], ascending = False)
    df.to_csv(f'scores_{base}_{use_view_based_pormpting}.csv', index=False) 


def main():
    find_file_with_pattern(base="vsd")
    


if __name__ == "__main__":
    main()