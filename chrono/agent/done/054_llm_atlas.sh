# TIMEOUT=300
# Advisor: the atlas must cover the LLMs (Qwen, Llama, OLMo, gpt-oss), and the
# SUCCESS case (entity names) alongside the failure case (documents).
J=$(sbatch --parsable --array=3 chrono/sbatch/C23_wb_akk.sbatch); echo "C23 olmo arm: $J"
G=$(sbatch --parsable chrono/sbatch/C23b_gptoss.sbatch); echo "C23b gpt-oss: $G"
E=$(sbatch --parsable chrono/sbatch/C26_deep_extract.sbatch); echo "C26 deep extract: $E"
A=$(sbatch --parsable --dependency=afterany:"${E%%;*}" chrono/sbatch/C25_atlas.sbatch); echo "C25 atlas v2 (after C26): $A"
