import numpy as np
import pandas as pd
import tqdm
import os
import math

def estimate_nodes(equation, max_d):
    cnt = 0
    total_sum = 0
    while 1:
        cnt += 1
        sum = (cnt*cnt*equation[0]+ cnt*equation[1] + equation[2])
        if ((equation[0] >= 0 and cnt > max_d) or cnt > (2*max_d)):
            break
        if (sum < 0):
            continue
        total_sum += sum
    return (total_sum)

def main():
    for planner in ["gbfs_hff", "astar_hff", "astar_lmcut", "gbfs_lmcut"]:
        main_path = "/data/preprocessing/20/"+planner
        ready_data_path = "/data/ready_data/40/"+planner
        domains =[
            ("airport", 30),
            ("blocks", 35),
            ("depot", 22),
            ("elevators", 30),
            ("freecell", 20),
            ("gripper", 20),
            ("logistics", 28),
            ("miconic", 30),
            ("movie", 30),
            ("openstacks", 30),
            ("parcprinter", 30),
            ("pegsol", 30),
            ("psr-small", 50),
            ("rovers", 30),
            ("satellite", 20),
            ("scanalyzer", 30),
            ("sokoban", 30),
            ("tpp", 30),
            ("transport", 30),
            ("woodworking", 30),
            ("zenotravel", 20),
        ]
        

        paths = []
        for domain in domains:
            temp_domain = domain[0]
            num_instances = domain[1]
            for instance in range(num_instances):
                if (instance<9):
                    str_instance = "0"+str(instance+1)
                else:
                    str_instance = str(instance+1)
                paths.append((main_path+"/"+temp_domain+"/task"+str_instance+".pddl.csv", ready_data_path+"/"+temp_domain+"_"+str_instance+"_x_.pt"))

        for path_tuple in paths:
            path = path_tuple[0]
            if os.path.isfile(os.getcwd()+path_tuple[1]) == False:
                continue
            column = ["level_1#", "level_1_H", "node_max"]
            df = pd.read_csv(os.getcwd()+path, skip_blank_lines=True, header=0, usecols=column)
            df = df.to_numpy()
            counter = {}
            estimation = []
            y_true = []
            print(len(df))
            for i in tqdm.tqdm(range(len(df))):
                temp = df[i][1]
                if math.isnan(temp):
                    continue
                if int(temp) in counter: 
                    counter[int(temp)] += 1
                else:
                    counter[int(temp)] = 1
                x = []
                y = []
                x.append(0)
                y.append(0)
                for j in counter:
                    x.append(j)
                    y.append(counter[j])
                if(len(x) < 2):
                    continue
                equation = np.polyfit(x, y, deg=2)
                if(equation[0] == 0):
                    print("matan")
                estimation.append((i+1)/(estimate_nodes(equation, max(counter)))*100)
                y_true.append(((df[i][0]/df[i][2]*100) - estimation[i])*((df[i][0]/df[i][2]*100) - estimation[i]))
            x_df = pd.DataFrame(estimation)
            words = path.split("/")
            x_df.to_csv("DBP/"+planner+"_"+words[-2]+"-"+words[-1])
            y_df = pd.DataFrame(y_true)
            y_df.to_csv("DBP/"+planner+"_y_"+words[-2]+"-"+words[-1])
if __name__ == '__main__':
    main()


 
    
