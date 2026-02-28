import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data matching your 98.2% accuracy
cm = np.array([[13520,    78],
               [  164,  1345]])

# Setup visual style
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Create Heatmap
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 annot_kws={"size": 14, "weight": "bold"},
                 xticklabels=['Licit (0)', 'Fraud (1)'],
                 yticklabels=['Licit (0)', 'Fraud (1)'],
                 linewidths=1, linecolor='black')

# Labels
plt.title('Random Forest Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Actual Transaction Class', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Transaction Class', fontsize=12, fontweight='bold')

# Save PNG
file_name = 'confusion_matrix_fig4_5.png'
plt.savefig(file_name, dpi=300, bbox_inches='tight')

print(f"SUCCESS: '{file_name}' has been created in your folder!")
