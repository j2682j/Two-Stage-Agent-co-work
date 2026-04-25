"""
BFCL 官方評估工具整合模組

封裝BFCL官方評估工具的呼叫，提供便捷的介面
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os


class BFCLIntegration:
    """BFCL官方評估工具整合類
    
    提供以下功能：
    1. 檢查BFCL評估工具是否已安裝
    2. 安裝BFCL評估工具
    3. 執行BFCL官方評估
    4. 解析評估結果
    
    使用範例：
        integration = BFCLIntegration()
        
        # 檢查並安裝
        if not integration.is_installed():
            integration.install()
        
        # 執行評估
        integration.run_evaluation(
            model_name="HelloAgents",
            category="simple_python",
            result_file="result/HelloAgents/BFCL_v3_simple_python_result.json"
        )
        
        # 解析結果
        scores = integration.parse_results(
            model_name="HelloAgents",
            category="simple_python"
        )
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """初始化BFCL集成
        
        Args:
            project_root: BFCL項目根目錄，如果為None則使用目前目錄
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.result_dir = self.project_root / "result"
        self.score_dir = self.project_root / "score"
    
    def is_installed(self) -> bool:
        """檢查BFCL評估工具是否已安裝
        
        Returns:
            True如果已安裝，False否則
        """
        try:
            result = subprocess.run(
                ["bfcl", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install(self) -> bool:
        """安裝BFCL評估工具
        
        Returns:
            True如果安裝成功，False否則
        """
        print("📦 正在安裝BFCL評估工具...")
        print("   執行: pip install bfcl-eval")
        
        try:
            result = subprocess.run(
                ["pip", "install", "bfcl-eval"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("[OK] BFCL評估工具安裝成功")
                return True
            else:
                print(f"[ERROR] 安裝失敗: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("[ERROR] 安裝逾時")
            return False
        except Exception as e:
            print(f"[ERROR] 安裝出錯: {e}")
            return False
    
    def prepare_result_file(
        self,
        source_file: Union[str, Path],
        model_name: str,
        category: str
    ) -> Path:
        """準備BFCL評估所需的結果檔案
        
        BFCL期望的檔案路徑格式：
        result/{model_name}/BFCL_v3_{category}_result.json
        
        Args:
            source_file: 源結果檔案路徑
            model_name: 模型名稱
            category: 評估類別
            
        Returns:
            目標檔案路徑
        """
        source_file = Path(source_file)
        
        # 建立目標目錄
        target_dir = self.result_dir / model_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 確定目標檔名
        target_file = target_dir / f"BFCL_v3_{category}_result.json"
        
        # 復制檔案
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, target_file)
            print(f"[OK] 結果檔案已準備")
            print(f"   源檔案: {source_file}")
            print(f"   目標檔案: {target_file}")
        else:
            print(f"[WARN] 源檔案不存在: {source_file}")
        
        return target_file
    
    def run_evaluation(
        self,
        model_name: str,
        category: str,
        result_file: Optional[Union[str, Path]] = None
    ) -> bool:
        """執行BFCL官方評估
        
        Args:
            model_name: 模型名稱
            category: 評估類別
            result_file: 結果檔案路徑（可選，如果提供則先準備檔案）
            
        Returns:
            True如果評估成功，False否則
        """
        # 如果提供了結果檔案，先準備
        if result_file:
            self.prepare_result_file(result_file, model_name, category)
        
        # 設定環境變數
        env = os.environ.copy()
        env["BFCL_PROJECT_ROOT"] = str(self.project_root)
        
        print(f"\n🔧 執行BFCL官方評估...")
        print(f"   模型: {model_name}")
        print(f"   類別: {category}")
        print(f"   項目根目錄: {self.project_root}")
        
        # 建構命令
        cmd = [
            "bfcl", "evaluate",
            "--model", model_name,
            "--test-category", category
        ]
        
        print(f"   命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env
            )
            
            if result.returncode == 0:
                print("[OK] BFCL評估完成")
                print(result.stdout)
                return True
            else:
                print(f"[ERROR] 評估失敗")
                print(f"   錯誤訊息: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("[ERROR] 評估逾時")
            return False
        except Exception as e:
            print(f"[ERROR] 評估出錯: {e}")
            return False
    
    def parse_results(
        self,
        model_name: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """解析BFCL評估結果
        
        Args:
            model_name: 模型名稱
            category: 評估類別
            
        Returns:
            評估結果字典，如果檔案不存在則回傳None
        """
        # BFCL評估結果路徑
        score_file = self.score_dir / model_name / f"BFCL_v3_{category}_score.json"
        
        if not score_file.exists():
            print(f"[WARN] 評估結果檔案不存在: {score_file}")
            return None
        
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"\n[INFO] BFCL評估結果")
            print(f"   模型: {model_name}")
            print(f"   類別: {category}")
            
            # 提取關鍵指標
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] 解析結果失敗: {e}")
            return None
    
    def get_summary_csv(self) -> Optional[Path]:
        """取得匯總CSV檔案路徑
        
        BFCL會生成以下CSV檔案：
        - data_overall.csv: 總體評分
        - data_live.csv: Live資料集評分
        - data_non_live.csv: Non-Live資料集評分
        - data_multi_turn.csv: 多輪對話評分
        
        Returns:
            data_overall.csv的路徑，如果不存在則回傳None
        """
        csv_file = self.score_dir / "data_overall.csv"
        
        if csv_file.exists():
            print(f"\n📄 匯總CSV檔案: {csv_file}")
            return csv_file
        else:
            print(f"[WARN] 匯總CSV檔案不存在: {csv_file}")
            return None
    
    def print_usage_guide(self):
        """打印使用指南"""
        print("\n" + "="*60)
        print("BFCL官方評估工具使用指南")
        print("="*60)
        print("\n1. 安裝BFCL評估工具：")
        print("   pip install bfcl-eval")
        print("\n2. 設定環境變數：")
        print(f"   export BFCL_PROJECT_ROOT={self.project_root}")
        print("\n3. 準備結果檔案：")
        print("   將評估結果放在: result/{model_name}/BFCL_v3_{category}_result.json")
        print("\n4. 執行評估：")
        print("   bfcl evaluate --model {model_name} --test-category {category}")
        print("\n5. 查看結果：")
        print("   評估結果在: score/{model_name}/BFCL_v3_{category}_score.json")
        print("   匯總結果在: score/data_overall.csv")
        print("\n" + "="*60)


