"""从 results.json 生成答辩用图表（matplotlib）。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
res = R["results"]
names = ["基线\n单色阈值", "改进\n参考差分", "改进+判定层"]
colors = ["#94a3b8", "#f59e0b", "#2563eb"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 图1：召回率 + 板级准确率
fig, ax = plt.subplots(figsize=(8, 4.6))
x = range(3)
rec = [r["recall"] * 100 for r in res]
acc = [r["board_acc"] * 100 for r in res]
w = 0.34
b1 = ax.bar([i - w/2 for i in x], rec, w, label="缺陷召回率", color=colors)
b2 = ax.bar([i + w/2 for i in x], acc, w, label="板级准确率", color=["#cbd5e1", "#fcd34d", "#60a5fa"])
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f"{b.get_height():.1f}",
            ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("百分比 (%)"); ax.set_ylim(0, 112)
ax.set_title("三档对照：缺陷召回率与板级判定准确率", fontsize=12)
ax.legend(loc="upper left"); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig1_compare.png"), dpi=150)
plt.close(fig)

# 图2：误判板数量（漏检板 + 误报板）
fig, ax = plt.subplots(figsize=(8, 4.6))
fn = [r["fn_board"] for r in res]
fp = [r["fp_board"] for r in res]
ax.bar(list(x), fn, .55, label="漏检板 (FN)", color="#ef4444")
ax.bar(list(x), fp, .55, bottom=fn, label="误报板/过杀 (FP)", color="#f59e0b")
for i in x:
    ax.text(i, fn[i]+fp[i]+1.2, f"合计 {fn[i]+fp[i]}", ha="center", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("误判板数量"); ax.set_title("板级误判拆解：漏检 vs 过杀", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig2_errors.png"), dpi=150)
plt.close(fig)

# 图3：各类召回率热力条形
fig, ax = plt.subplots(figsize=(8, 4.2))
classes = list(res[2]["per_class_recall"].keys())
cn = ["漏孔","鼠咬","开路","短路","毛刺","杂铜"]
base_r = [res[0]["per_class_recall"][c]*100 for c in classes]
imp_r = [res[2]["per_class_recall"][c]*100 for c in classes]
xx = range(len(classes))
ax.bar([i-.2 for i in xx], base_r, .4, label="基线", color="#94a3b8")
ax.bar([i+.2 for i in xx], imp_r, .4, label="改进+判定层", color="#2563eb")
ax.set_xticks(list(xx)); ax.set_xticklabels(cn, fontsize=11)
ax.set_ylabel("召回率 (%)"); ax.set_ylim(0,112)
ax.set_title("分缺陷类型召回率：缺口型缺陷是基线盲区", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig3_per_class.png"), dpi=150)
plt.close(fig)

print("figures saved")
