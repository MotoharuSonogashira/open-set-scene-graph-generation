# Recalls
utils/recalls.py -m R -p2 output/open-outer/{freq,imp,vctree}{,-.[0-9]}-sgcls
utils/recalls.py -m R -p2 output/open-outer/{freq,imp,vctree}{,-.[0-9]}

# Counts
(for x (output/open-outer/{freq,imp,vctree}-.[0-9]) @ utils/counts.py -C $x -O $x/eval)
utils/plot_counts.py output/open-outer/{freq,imp,vctree}-{.0,.[1-9]} -n Freq Freq+ IMP IMP+ VCTree VCTree+ --figsize='(6,6)' -O '{}_counts'.pdf

# Visualization
@ utils/viz.py output/open-outer/vctree-.[1-9] -O output/open-outer/gt/viz -GgU -m100
for x (output/open-outer/vctree-.[1-9]) @ utils/viz.py $x -O $x/viz -Ug -m100
