/**
 * RuAG演示主脚本
 * 初始化演示环境和事件处理
 */
document.addEventListener('DOMContentLoaded', function() {
    // 设置默认查询文本
    if (document.getElementById('input-query').value === '') {
        const defaultQueries = {
            'relation': '分析文本中梅西的俱乐部关系',
            'anomaly': '检测以下日志序列是否存在异常：["系统启动", "数据库连接失败", "重试连接", "连接成功"]'
        };
        
        document.getElementById('input-query').value = defaultQueries['relation'];
        
        // 标签切换时更新默认查询
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                document.getElementById('input-query').value = defaultQueries[tab] || '';
            });
        });
    }
});