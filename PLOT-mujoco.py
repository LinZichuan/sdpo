# 导入库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
# import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
seed_num = 10
env = "Humanoid-v2"
exptype = 'sd_espo'

if exptype == 'one_side':
    method_list = ['SD-ESPO', 'RSD-ESPO', 'LSD-ESPO']
elif exptype == 'sd_espo':
    method_list = ['ESPO', 'SD-ESPO']

basedir = [
    f"log/{env}.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean-",
    f"log_espo_cd/{env}.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25-",
    f"log_espo_cd_unbias/{env}.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25.unbias_True-",
    f"log_espo_cd_value/{env}.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25.sd_value_True-",
    f"log_espo_cd_rminus/Humanoid-v2.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25.rminus_True-",
    f"log_espo_cd_noabs_rminus/Humanoid-v2.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25.no_abs_True.rminus_True-",
    f"log_espo_cd_noabs/{env}.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25.no_abs_True-",
    f"log_espo_cd_lclip/{env}.ppo2.dropout.worker_1.norm_obs.norm_rew.model_type_dropout.tv_threshold_0.25.reduce_type_reduce_mean.use_cd_True.sd_delta_0.25.lclip_True-",
]
NUM = len(basedir)
label = ['ESPO', 'SD-ESPO', 'SD-ESPO-unbias(1/M,M)', 'SD-ESPO-value', 'SD-ESPO-rminus', 'SD-ESPO(no abs)-rminus', 'RSD-ESPO', 'LSD-ESPO']
label = label[:NUM]
plt.figure(figsize=(15, 10))

T = sys.argv[1]

data = []
for j in range(NUM):
    check_dir = basedir[j].split('/')[0]
    if label[j] not in method_list: continue
    if not os.path.exists(check_dir): continue
    df = pd.DataFrame([])
    for i in range(seed_num):
        df1 = pd.read_csv(os.path.join(basedir[j] + str(i), "progress.csv"))
        #if j == 0:
        #    df1["loss/obj_std"] = df1["loss/ori_obj_std"]
        #else:
        #    df1["loss/obj_std"] = df1["loss/sd_obj_std"]
        df1["loss/ratio_max"] = np.log(df1["loss/ratio_max"])
        df1["loss/ratio_min"] = np.log(df1["loss/ratio_min"])
        df1["loss/ratio_mean"] = df1["loss/ratio_mean"] - 1.0
        print (df1["loss/obj_std"])
        print('df1: len {0}; last:{1}'.format(len(df1), df1.iloc[-1].tolist()))
        #df1['loss/train_std'] = df1['loss/train_std'].map(lambda x: np.exp(x))
        #print("df1:", df1.iloc[-1].tolist())
        df=df.append(df1)
    df.index = range(len(df))

    print (df)
    # label = ['ESPO','SD-ESPO-0.15','SD-ESPO-0.25','SD-ESPO-0.3','SD-ESPO-0.4','CD-TRPO']
    df.insert(len(df.columns), "algo", label[j])

    # d5f.insert(len(d5f.columns), "algo", label[5])
    data.append(df)


print ('start painting')
# #
# data.append(d4f)
# data.append(d5f)
data = pd.concat(data, ignore_index=True)
data.index = range(len(data))
#print(df)
# 设置图片大小
# 画图
if T == "var":
    #sns.lineplot(data=data,x="misc/total_timesteps", y="loss/ori_obj_std",hue="algo", style="algo",ci=None)
    sns.lineplot(data=data,x="misc/total_timesteps", y="loss/obj_std",hue="algo", style="algo", ci=None)
if T == "rew":
    sns.lineplot(data=data,x="misc/total_timesteps", y="eprewmean",hue="algo", style="algo", ci=None)
if T == "ratio":
    sns.lineplot(data=data,x="misc/total_timesteps", y="loss/ratio_max",hue="algo", style="algo", ci=None)
    sns.lineplot(data=data,x="misc/total_timesteps", y="loss/ratio_min",hue="algo", style="algo", ci=None)
if T == "ratio_mean":
    sns.lineplot(data=data,x="misc/total_timesteps", y="loss/ratio_mean",hue="algo", style="algo", ci=None)
if T == "ratio_dev":
    sns.lineplot(data=data,x="misc/total_timesteps", y="",hue="algo", style="algo", ci=None)
#sns.lineplot(data=data,x="misc/total_timesteps", y="loss/ori_obj_std",hue="algo", style="algo",ci=None, color=color[j])
#sns.lineplot(data=data,x="misc/total_timesteps", y="eprewmean",hue="algo", style="algo")
#sns.lineplot(data=data,x="misc/total_timesteps", y="loss/value_loss",hue="algo", style="algo")

xscale = np.max(data["misc/total_timesteps"]) > 5e3
if xscale:
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.legend(fontsize=25)

plt.xticks(size=20)
plt.yticks(size=20)

plt.xlabel("Timesteps", size=28)
if T == "var":
    plt.ylabel("Variance of surrogate objective", size=28)
if T == "rew":
    plt.ylabel("Returns", size=28)
if T == "ratio":
    plt.ylabel("Ratio range (log)", size=28)
if T == "ratio_mean":
    plt.ylabel("Average ratio bias", size=28)
if T == "ratio_dev":
    plt.ylabel("Ratio deviation", size=28)
plt.title(env, size=28)
plt.show()
plt.savefig(f'{exptype}_{env}_plot_{T}.png')
