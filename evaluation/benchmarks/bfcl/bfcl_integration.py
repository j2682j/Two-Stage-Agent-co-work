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
    """
    負責在 evaluation.benchmarks.bfcl.bfcl_integration 中封裝 BFCLIntegration，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        project_root: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        負責執行 BFCLIntegration 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            project_root: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.result_dir = self.project_root / "result"
        self.score_dir = self.project_root / "score"
    
    def is_installed(self) -> bool:
        """
        負責執行 BFCLIntegration 中的 is_installed 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLIntegration 中的 install 流程，依照 BFCLIntegration 的流程需求處理 install 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLIntegration 中的 prepare_result_file 流程，依照 BFCLIntegration 的流程需求處理 prepare_result_file 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            source_file: 評估、推理或工具執行後產生的結果與分數資料。
            model_name: 評估、推理或工具執行後產生的結果與分數資料。
            category: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Path。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLIntegration 中的 run_evaluation 流程，依照 BFCLIntegration 的流程需求處理 run_evaluation 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            category: 此流程需要使用的輸入資料。
            result_file: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLIntegration 中的 parse_results 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            category: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLIntegration 中的 get_summary_csv 流程，依照 BFCLIntegration 的流程需求處理 get_summary_csv 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Path]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        csv_file = self.score_dir / "data_overall.csv"
        
        if csv_file.exists():
            print(f"\n📄 匯總CSV檔案: {csv_file}")
            return csv_file
        else:
            print(f"[WARN] 匯總CSV檔案不存在: {csv_file}")
            return None
    
    def print_usage_guide(self):
        """
        負責執行 BFCLIntegration 中的 print_usage_guide 流程，依照 BFCLIntegration 的流程需求處理 print_usage_guide 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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


