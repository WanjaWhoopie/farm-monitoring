"""Generate a scientific summary figure from saved measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
baseline=json.loads((ROOT/'reports'/'acoustic_baseline.json').read_text())
counts=pd.read_csv(ROOT/'data'/'processed'/'vision'/'behavior_counts.csv').sort_values('pig_annotation_slots')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13.5,6.7),gridspec_kw={'width_ratios':[1,1.25]})
fig.patch.set_facecolor('white')
keys=['random_call_optimistic','team_date_group_proxy','leave_recording_team_out']
values=[100*baseline[k]['oof_balanced_accuracy'] for k in keys]
ax.barh([2,1,0],values,color=['#9aadb8','#39798a','#173f57'],height=0.55)
ax.set_yticks([2,1,0],['Random calls\n(optimistic)','Team/date\ngroups','Recording-team\nholdout'])
for y,value in zip([2,1,0],values):ax.text(value+1,y,f'{value:.1f}%',va='center',fontweight='bold',fontsize=12)
ax.axvline(50,color='#b85243',linestyle='--',linewidth=1.2)
ax.set_xlim(0,100);ax.set_xlabel('Balanced accuracy (%)');ax.set_ylim(-0.65,2.65)
ax.set_title('Audio performance changes with the split',loc='left',fontsize=14,fontweight='bold',pad=20)
ax.grid(axis='x',alpha=0.12);ax.set_axisbelow(True)
ax.text(0,-0.22,'Logistic regression; 6,150 calls, 10 supplied acoustic features.\n737 calls excluded because all 10 features are missing.\nScores pool held-out predictions; no hyperparameter tuning.',transform=ax.transAxes,fontsize=9,va='top',linespacing=1.5,color='#444444')
bx.barh(counts['behavior'],counts['pig_annotation_slots'],color='#39798a',height=0.65)
bx.set_xscale('log');bx.set_xlim(0.7,60000)
bx.set_xlabel('Derived pig-frame annotation slots (log scale)')
for i,value in enumerate(counts['pig_annotation_slots']):bx.text(value*1.12,i,f'{value:,}',va='center',fontsize=9)
bx.set_title('Some video behaviors have almost no labels',loc='left',fontsize=14,fontweight='bold',pad=20)
bx.grid(axis='x',alpha=0.12);bx.set_axisbelow(True)
bx.text(0,-0.22,'12 clips, 7 days, one pen; 57,600 pig-frame slots.\nBoxes interpolated; behavior carried forward from source changes.\nSlots are correlated observations, not independent animals.',transform=bx.transAxes,fontsize=9,va='top',linespacing=1.5,color='#444444')
fig.suptitle('AI4E data probe: design evaluation before scaling up',x=0.04,y=0.99,ha='left',fontsize=18,fontweight='bold',color='#173f57')
fig.subplots_adjust(left=0.12,right=0.96,top=0.86,bottom=0.24,wspace=0.72)
destination=ROOT/'reports'/'data_probe_findings.png'
fig.savefig(destination,dpi=150,facecolor='white')
plt.close(fig)
print(destination)
