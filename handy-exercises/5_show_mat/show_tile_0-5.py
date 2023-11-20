import matplotlib.pyplot as plt
import numpy as np
X=np.zeros((4096*4096,1))

# NOTE: 1. 在循环最深处为矩阵赋值
vals = np.linspace(0,1,256)
np.random.shuffle(vals)
# cmap = plt.cm.colors.ListedColormap(plt.cm.Reds(vals))
cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
# cmap = plt.cm.Reds

blockIdx_x = 0
threadIdx_x = 0
threadIdx_x_1 = threadIdx_x
threadIdx_x_2 = threadIdx_x_1
p0_shared_1 = np.zeros((256, 1))
p1_shared_1 = np.zeros((256, 1))
T_matmul_NT_local_1 = np.zeros((8, 1))
T_matmul_NT = np.zeros((4096*4096, 1)) # TODO:
p0 = np.zeros((4096*4096, 1))	# TODO:
p1 = np.zeros((4096*4096, 1))	# TODO:

def floordiv(x,y):
	return x // y
def floormod(x,y):
	return x - y * (x // y)

cnt = 1
			
for blockIdx_x in range(2048):
	for threadIdx_x in range(128):
		threadIdx_x_1 = threadIdx_x
		threadIdx_x_2 = threadIdx_x_1

		for i_outer_outer_inner_j_outer_outer_inner_fused in range(0, 8):
			# for i_c_outer_inner_init in range(0, 8):
			# 	T_matmul_NT_local_1[i_c_outer_inner_init] = 0x0f32
			for k_outer_outer in range(0, 512):
				for ax0_ax1_fused_inner in range(0, 4):
					if threadIdx_x_1 < 64:
						# p0_shared_1[((threadIdx_x_1*4) + ax0_ax1_fused_inner)] = p0[((((((floordiv(blockIdx_x, 32)*262144) + (floordiv(i_outer_outer_inner_j_outer_outer_inner_fused, 4)*131072)) + (floordiv(threadIdx_x_1, 2)*4096)) + (k_outer_outer*8)) + (floormod(threadIdx_x_1, 2)*4)) + ax0_ax1_fused_inner)]
						p0[((((((floordiv(blockIdx_x, 32)*262144) + (floordiv(i_outer_outer_inner_j_outer_outer_inner_fused, 4)*131072)) + (floordiv(threadIdx_x_1, 2)*4096)) + (k_outer_outer*8)) + (floormod(threadIdx_x_1, 2)*4)) + ax0_ax1_fused_inner)] = cnt*1
				# for ax0_ax1_fused_inner_1 in range(0, 4):
				# 	if threadIdx_x_2 < 64:
				# 		p1_shared_1[((threadIdx_x_2*4) + ax0_ax1_fused_inner_1)] = p1[((((((floormod(blockIdx_x, 32)*524288) + (floormod(i_outer_outer_inner_j_outer_outer_inner_fused, 4)*131072)) + (floordiv(threadIdx_x_2, 2)*4096)) + (k_outer_outer*8)) + (floormod(threadIdx_x_2, 2)*4)) + ax0_ax1_fused_inner_1)]
				
				# for k_outer_inner in range(0, 4):
				# 	for i_c_outer_inner in range(0, 8):
				# 		for k_inner in range(0, 2) :
				# 			cse_var_1 = (k_outer_inner*2)
				# 			T_matmul_NT_local_1[i_c_outer_inner] = (T_matmul_NT_local_1[i_c_outer_inner] + (p0_shared_1[((((floordiv(threadIdx_x, 32)*64) + (i_c_outer_inner*8)) + cse_var_1) + k_inner)]*p1_shared_1[(((floormod(threadIdx_x, 32)*8) + cse_var_1) + k_inner)]))
			# for i_inner in range(0, 8):
			# 	T_matmul_NT[(((((((floordiv(blockIdx_x, 32)*262144) + (floordiv(i_outer_outer_inner_j_outer_outer_inner_fused, 4)*131072)) + (floordiv(threadIdx_x, 32)*32768)) + (i_inner*4096)) + (floormod(blockIdx_x, 32)*128)) + (floormod(i_outer_outer_inner_j_outer_outer_inner_fused, 4)*32)) + floormod(threadIdx_x, 32))] = T_matmul_NT_local_1[i_inner]

		bound = 256 
		id = 0
		msg = 'blockIdx_x'
		title = 'show_tile_m0_p0'
		# 不要修改下面内容
		if cnt > bound:
			break
		elif cnt == bound:
			plt.matshow(p0.reshape(4096,4096)[:512,:512], cmap=cmap)
			plt.title("%s : %s"%(title, msg))
			plt.savefig('%s_%d_%s.png'%(title, id, msg), format='png', dpi=2048)
			print('cnt:%d'%(cnt))
		cnt+=1

