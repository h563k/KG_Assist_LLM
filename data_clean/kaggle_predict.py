import numpy as np
from scipy import stats
mbti_data = [(['87.50%', '100.00%', '92.50%', '85.00%'], 0.9090064516129032,40),
(['85.00%', '97.50%', '95.00%', '77.50%'], 0.880075272694492,40),
(['80.00%', '90.00%', '88.00%', '78.00%'], 0.8369030684820159,50),
(['92.00%', '96.00%', '90.00%', '72.00%'], np.float64(0.864187866927593),50),
(['86.00%', '92.00%', '84.00%', '80.00%'], np.float64(0.852837839571918),50) ,
 (['92.00%', '96.00%', '86.00%', '82.00%'], np.float64(0.8867358781755952),50)]


full = np.zeros(4)
nums = 0
for mbti in mbti_data:
    value,_,num = mbti
    nums += num
    value =[float(x.strip('%'))*num/100 for x in value]
    value = np.array(value).astype(int)
    full += value
print(full, nums)
full = full/nums
print([f"{float(x):.2%}" for x in full])
print(f"调和平均为: {(float(stats.hmean(full))):.2%}")

"""
[244. 266. 249. 221.] 280
['87.14%', '95.00%', '88.93%', '78.93%']
调和平均为: 87.11%
"""