import matplotlib.pyplot as plt
import numpy as np
X=np.zeros((4096*4096,1))

# NOTE: 1. 在循环最深处为矩阵赋值
cnt = 1
for i_outer_outer_outer_outer in range(0, 64):
    for j_outer_outer_outer_outer in range(0, 32):
        for j_outer_outer_outer_inner in range(0, 4):
            for i_outer_outer_inner in range(0, 4):
                for j_outer_outer_inner in range(0, 32):
                    for i_outer_inner in range(0, 8):
                        for i_inner in range(0, 2):
                            X[(((((((i_outer_outer_outer_outer*262144) + (i_outer_outer_inner*65536)) + (i_outer_inner*8192)) + (i_inner*4096)) + (j_outer_outer_outer_outer*128)) + (j_outer_outer_outer_inner*32)) + j_outer_outer_inner)] = cnt*1
                            # cse_var_3 = ((((i_outer_outer_outer_outer*262144) + (i_outer_outer_inner*65536)) + (i_outer_inner*8192)) + (i_inner*4096))
                            # cse_var_2 = (k_outer_outer*8)
                            # cse_var_1 = (((cse_var_3 + (j_outer_outer_outer_outer*128)) + (j_outer_outer_outer_inner*32)) + j_outer_outer_inner)
                            # T_matmul_NT_local[cse_var_1] = (T_matmul_NT_local[cse_var_1] + (p0[((cse_var_3 + cse_var_2) + k_outer_inner)]*p1[(((((j_outer_outer_outer_outer*524288) + (j_outer_outer_outer_inner*131072)) + (j_outer_outer_inner*4096)) + cse_var_2) + k_outer_inner)]))                  
            
    # NOTE: 2. 把下面代码放到想要遍历的轴上，并修改参数
    bound = 64 
    id = 3
    msg = 'i_outer_outer_outer_outer'
    title = ''
    # 不要修改下面内容
    if cnt > bound:
        break
    elif cnt == bound:
        plt.matshow(X.reshape(4096,4096), cmap=plt.cm.Reds)
        plt.title("show tile : %s"%(msg))
        plt.savefig('%s_show_tile_%d_%s.png'%(title, id, msg), format='png', dpi=2048)
    cnt+=1