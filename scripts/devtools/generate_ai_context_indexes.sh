#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/docs/ai_context"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

GENERATED_AT="$(date -Iseconds)"

TREE_OUT="${OUT_DIR}/02_DIRECTORY_TREE.txt"
TASK_OUT="${OUT_DIR}/04_TASK_REGISTRY_INDEX.md"
SYMBOL_OUT="${OUT_DIR}/11_SYMBOL_INDEX_RAW.txt"

EXCLUDES=(
  ".git"
  "__pycache__"
  ".pytest_cache"
  ".mypy_cache"
  ".ruff_cache"
  "*.egg-info"
  ".venv"
  "env"
  "logs"
  "runs"
  "wandb"
)

write_tree_with_find() {
  {
    echo "# Directory tree for ${REPO_ROOT}"
    echo "# Generated at: ${GENERATED_AT}"
    echo "# Source command: find fallback because tree is not installed"
    echo
    find . \
      -path "./.git" -prune -o \
      -path "*/__pycache__" -prune -o \
      -path "*/.pytest_cache" -prune -o \
      -path "*/.mypy_cache" -prune -o \
      -path "*/.ruff_cache" -prune -o \
      -path "./logs" -prune -o \
      -path "./runs" -prune -o \
      -path "./wandb" -prune -o \
      -print \
      | LC_ALL=C sort \
      | awk '
          BEGIN { print "." }
          NR == 1 { next }
          {
            path = $0
            sub(/^\.\//, "", path)
            n = split(path, parts, "/")
            indent = ""
            for (i = 1; i < n; i++) {
              indent = indent "  "
            }
            print indent "- " parts[n]
          }
        '
  } > "${TREE_OUT}"
}

if command -v tree >/dev/null 2>&1; then
  {
    echo "# Directory tree for ${REPO_ROOT}"
    echo "# Generated at: ${GENERATED_AT}"
    echo "# Source command: tree"
    echo
    tree -a -I "$(IFS='|'; echo "${EXCLUDES[*]}")" --dirsfirst .
  } > "${TREE_OUT}"
else
  write_tree_with_find
fi

if ! command -v rg >/dev/null 2>&1; then
  {
    echo "# 任务注册索引"
    echo
    echo "生成失败：当前环境没有安装 ripgrep（rg）。请安装 ripgrep 后运行："
    echo
    echo '```bash'
    echo "bash scripts/devtools/generate_ai_context_indexes.sh"
    echo '```'
  } > "${TASK_OUT}"
  {
    echo "# Symbol index"
    echo
    echo "生成失败：当前环境没有安装 ripgrep（rg）。请安装 ripgrep 后重新生成。"
  } > "${SYMBOL_OUT}"
  exit 0
fi

{
  echo "# 任务注册索引"
  echo
  echo "- 生成时间：${GENERATED_AT}"
  echo "- 仓库根目录：${REPO_ROOT}"
  echo "- 检索范围：source/ scripts/"
  echo
  echo "## gym.register 附近代码"
  echo
  echo '```text'
  rg -n -C 8 --glob '*.py' 'gym\.register' source scripts || true
  echo '```'
  echo
  echo "## Entry point 和配置入口检索"
  echo
  echo '```text'
  rg -n --glob '*.py' 'entry_point|env_cfg_entry_point|rsl_rl_cfg_entry_point|skrl_cfg_entry_point|rl_games_cfg_entry_point|agents\.__name__|__name__' source scripts || true
  echo '```'
  echo
  echo "## 任务导入链检索"
  echo
  echo '```text'
  rg -n --glob '*.py' 'import_packages|huilun_isaaclab\.tasks|gym\.registry|Template-' source scripts || true
  echo '```'
} > "${TASK_OUT}"

{
  echo "# Symbol index raw"
  echo
  echo "- Generated at: ${GENERATED_AT}"
  echo "- Scope: source/ scripts/"
  echo
  echo "## Python classes and functions"
  echo
  echo '```text'
  rg -n --glob '*.py' '^(class|def) [A-Za-z_][A-Za-z0-9_]*|^    class [A-Za-z_][A-Za-z0-9_]*|^    def [A-Za-z_][A-Za-z0-9_]*' source scripts || true
  echo '```'
  echo
  echo "## Isaac Lab config decorators and config classes"
  echo
  echo '```text'
  rg -n --glob '*.py' '@configclass|Cfg\)|Cfg:|ManagerBasedRLEnvCfg|DirectRLEnvCfg|DirectMARLEnvCfg|InteractiveSceneCfg|ArticulationCfg|AssetBaseCfg|RslRl' source scripts || true
  echo '```'
  echo
  echo "## Manager, MDP, and RL terms"
  echo
  echo '```text'
  rg -n --glob '*.py' 'ObservationTermCfg|ObservationGroupCfg|RewardTermCfg|TerminationTermCfg|EventTermCfg|SceneEntityCfg|ActionCfg|JointEffortActionCfg|gym\.make|RslRlVecEnvWrapper|OnPolicyRunner|DistillationRunner|hydra_task_config|AppLauncher' source scripts || true
  echo '```'
  echo
  echo "## Asset and resource references"
  echo
  echo '```text'
  rg -n --glob '*.py' --glob '*.toml' --glob '*.yaml' --glob '*.urdf' --glob '*.xml' 'resources/|robots/|l5a|urdf|usd|xml|mesh|STL|ArticulationCfg|CARTPOLE_CFG' source scripts resources || true
  echo '```'
} > "${SYMBOL_OUT}"

echo "Generated:"
echo "  ${TREE_OUT}"
echo "  ${TASK_OUT}"
echo "  ${SYMBOL_OUT}"
