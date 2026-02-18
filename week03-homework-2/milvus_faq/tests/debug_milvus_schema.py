"""Milvus Schema 诊断与调试工具。

此脚本用于检查 Milvus 集合的架构，并自动处理向量维度不匹配的问题。
"""

import sys
import os

# 为了支持 IDE 直接点击运行按钮，将项目根目录加入 sys.path
# 确保在 milvus_faq 包外部运行也能正确识别包路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pymilvus import connections, utility, Collection
from milvus_faq.config import settings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

def check_collection_schema():
    """连接 Milvus 并检查指定集合的架构。
    
    如果发现数据库中的维度与配置文件中的维度不一致，将自动删除旧集合。
    """
    console.print(Panel.fit("[bold blue]Milvus Schema 诊断工具[/bold blue]", subtitle="v1.0"))
    
    try:
        # 建立数据库连接
        with console.status("[bold green]正在连接 Milvus...[/bold green]"):
            connections.connect(uri=settings.milvus.URI, token=settings.milvus.TOKEN)
        rprint(f"[green]✔[/green] 已成功连接至: [cyan]{settings.milvus.URI}[/cyan]")
        
        collection_name = settings.milvus.COLLECTION_NAME
        
        # 检查集合是否存在
        if not utility.has_collection(collection_name):
            rprint(f"[yellow]⚠[/yellow] 集合 [bold magenta]{collection_name}[/bold magenta] 不存在，无需检查。")
            return
            
        # 加载集合元数据
        collection = Collection(collection_name)
        
        # 构建富文本表格展示架构
        table = Table(title=f"集合 [bold magenta]{collection_name}[/bold magenta] 的字段架构", show_header=True, header_style="bold cyan")
        table.add_column("字段名称", style="dim")
        table.add_column("数据类型")
        table.add_column("关键参数")
        table.add_column("描述")

        for field in collection.schema.fields:
            # 101 是 FloatVector 的枚举值，代表浮点向量
            dtype_str = "FloatVector" if field.dtype == 101 else str(field.dtype)
            table.add_row(
                field.name,
                dtype_str,
                str(field.params) if field.params else "-",
                field.description or "-"
            )
        
        console.print(table)
        
        # 执行维度校验逻辑
        for field in collection.schema.fields:
            if field.dtype == 101:  # 仅针对向量字段
                current_dim = int(field.params.get('dim', 0))
                config_dim = settings.milvus.DIMENSION
                
                if current_dim != config_dim:
                    console.print(Panel(
                        f"[bold red]检测到维度不匹配！[/bold red]\n\n"
                        f"数据库当前维度: [yellow]{current_dim}[/yellow]\n"
                        f"代码配置要求维度: [green]{config_dim}[/green]\n\n"
                        f"[white]正在自动清理冲突的集合...[/white]",
                        title="错误诊断",
                        border_style="red"
                    ))
                    
                    # 删除不匹配的集合以便后续重建
                    utility.drop_collection(collection_name)
                    rprint(f"[bold green]✔[/bold green] 已成功删除集合 [bold magenta]{collection_name}[/bold magenta]。")
                    rprint("[bold yellow]请重新启动应用以使用正确的配置重建数据库。[/bold yellow]")
                else:
                    rprint(f"[bold green]✔[/bold green] 向量维度匹配成功: [cyan]{current_dim}[/cyan]")

    except Exception as e:
        console.print(Panel(
            f"[bold red]运行出错:[/bold red]\n{str(e)}",
            title="异常捕获",
            border_style="red"
        ))
        sys.exit(1)

if __name__ == "__main__":
    check_collection_schema()
