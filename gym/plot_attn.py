import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi']=600

folder_name = './DeMa/gym/step_attention'
start_idx = 0
end_idx = 28
# 330-390，240-300
with open(f'{folder_name}/all_attn_data.pkl', 'rb') as file:
    data_loaded = pickle.load(file)

attn_mat = data_loaded['step_attn_matrices']
print(attn_mat.shape)
attn_mat
attn_mat = attn_mat[:,start_idx:end_idx]
mean_attn_matrix = np.mean(attn_mat, axis=2)
cmap = 'turbo'

for selected_layer in range(3):
    p = mean_attn_matrix[selected_layer]
    plt.clf()
    y = np.linspace(start_idx, end_idx, end_idx-start_idx)
    x = np.linspace(0, 60, 60)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(y, x, p[:], cmap=cmap)
    ax.set_xlabel('Decision Timestep',fontsize=15)
    ax.set_ylabel('Input Sequence',fontsize=15)
    ax.set_zlabel('Score',fontsize=15)
    # plt.axis('off')
    # im = plt.imshow(p, interpolation='nearest', cmap=cmap)
    # plt.colorbar(im)
    plt.savefig(f'{folder_name}/step_attn_{selected_layer}.png', bbox_inches='tight', pad_inches=0.5, dpi=600)
    # plt.tight_layout()
    plt.show()
    print(f'step_attn_{selected_layer}.pdf saved successfully!')
mean_attn_matrix = np.mean(mean_attn_matrix, axis=0)
plt.clf()
y = np.linspace(start_idx, end_idx, end_idx-start_idx)
x = np.linspace(0, 60, 60)
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Decision Timestep',fontsize=15)
ax.set_ylabel('Input Sequence',fontsize=15)
ax.set_zlabel('Score',fontsize=15)
ax.plot_surface(y, x, mean_attn_matrix, cmap=cmap)
# im = plt.imshow(mean_attn_matrix, interpolation='nearest', cmap=cmap)
# plt.colorbar(im)
# plt.axis('off')
# plt.tight_layout()
plt.savefig(f'{folder_name}/step_attn_fused.png', bbox_inches='tight', pad_inches=0.3, dpi=600)
# plt.show()
print(f'step_attn_fused.pdf saved successfully!')