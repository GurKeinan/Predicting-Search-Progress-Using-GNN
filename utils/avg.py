import pandas as pd
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
    for planner in ["gbfs_hff"]:
        for method in ["pbp"]:
            main_path = "/"+method+"/"+planner+"_y_"
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
                    paths.append((main_path+temp_domain+"-task"+str_instance+".pddl.csv", ready_data_path+"/"+temp_domain+"_"+str_instance+"_x_.pt"))
                    

            avgs = []
            for path_tuple in paths:
                path = path_tuple[0]
                if (os.path.isfile(os.getcwd()+path)==False):
                    continue
                df = pd.read_csv(os.getcwd()+path, skip_blank_lines=True, header=0)
                df = df.to_numpy()
                sum = 0
                for i in range(len(df)):
                    if (math.isnan (df[i][1])):
                        sum+= 0
                    else:
                        sum += df[i][1]
                avgs.append((sum/len(df),path))
            x_df = pd.DataFrame(avgs)
            x_df.to_csv(method+"_"+planner+".csv")
if __name__ == '__main__':
    main()


 
    
