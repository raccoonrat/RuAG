document.addEventListener('DOMContentLoaded', function() {
    // 初始化Mermaid
    mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
    
    // 绘制处理流程图
    drawPipeline();
    
    // 绘制技术亮点可视化
    drawTechVisualizations();
    
    // 绑定事件处理
    bindEvents();
    
    // 加载示例数据
    loadExampleData();
});

// 绘制RuAG处理流程图
function drawPipeline() {
    const pipelineDiv = document.getElementById('pipeline-viz');
    
    const pipelineDiagram = `
    graph LR
        A[用户查询] --> B[数据处理]
        B --> C[规则学习]
        C --> D[规则翻译]
        D --> E[规则整合]
        E --> F[规则增强生成]
        F --> G[规则验证]
        G --> H[文本调整]
        H --> I[最终输出]
        
        style C fill:#ff7f50,stroke:#333,stroke-width:2px
        style F fill:#ff7f50,stroke:#333,stroke-width:2px
        style G fill:#ff7f50,stroke:#333,stroke-width:2px
    `;
    
    pipelineDiv.innerHTML = `<div class="mermaid">${pipelineDiagram}</div>`;
    mermaid.init(undefined, '.mermaid');
}

// 绘制技术亮点可视化
function drawTechVisualizations() {
    // MCTS可视化
    const mctsViz = document.getElementById('mcts-viz');
    const mctsDiagram = `
    graph TD
        R[根节点] --> A[选择]
        A --> B[扩展]
        B --> C[模拟]
        C --> D[回溯]
        D --> A
        
        style A fill:#ff7f50,stroke:#333
        style B fill:#ff7f50,stroke:#333
        style C fill:#ff7f50,stroke:#333
        style D fill:#ff7f50,stroke:#333
    `;
    mctsViz.innerHTML = `<div class="mermaid">${mctsDiagram}</div>`;
    
    // 谓词处理可视化
    const predicateViz = document.getElementById('predicate-viz');
    const predicateDiagram = `
    graph TD
        A[谓词定义] --> B[谓词空间搜索]
        B --> C[谓词组合优化]
        C --> D[规则评估]
        D --> E[规则筛选]
        
        style B fill:#ffd700,stroke:#333
        style C fill:#ffd700,stroke:#333
    `;
    predicateViz.innerHTML = `<div class="mermaid">${predicateDiagram}</div>`;
    
    // 规则整合可视化
    const integrationViz = document.getElementById('integration-viz');
    const integrationDiagram = `
    graph LR
        A[原始提示] --> B{规则整合}
        C[学习规则] --> B
        B --> D[增强提示]
        D --> E[LLM]
        E --> F[规则验证]
        F --> G[最终输出]
        
        style B fill:#4a6fa5,stroke:#333
        style F fill:#4a6fa5,stroke:#333
    `;
    integrationViz.innerHTML = `<div class="mermaid">${integrationDiagram}</div>`;
    
    // 初始化所有Mermaid图表
    mermaid.init(undefined, '.mermaid');
}

// 绑定事件处理
function bindEvents() {
    // 标签切换
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            tabBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // 加载对应任务的示例数据
            loadExampleData(this.dataset.tab);
        });
    });
    
    // 运行按钮
    const runBtn = document.getElementById('run-btn');
    runBtn.addEventListener('click', function() {
        runDemo();
    });
}

// 加载示例数据
function loadExampleData(task = 'relation') {
    const inputQuery = document.getElementById('input-query');
    
    if (task === 'relation') {
        inputQuery.value = '分析以下文本中的实体关系：\'梅西是巴塞罗那足球俱乐部的球员。\'';
        
        // 加载示例规则
        const rulesList = document.getElementById('rules-list');
        rulesList.innerHTML = `
            <div class="rule-item">
                <h4>规则 1</h4>
                <p>当实体A为哈兰德时，存在效力于关系（准确度：1.00）。</p>
            </div>
            <div class="rule-item">
                <h4>规则 2</h4>
                <p>当实体为姆巴佩时，可以确定其效力于关系，精确度为1.00。</p>
            </div>
            <div class="rule-item">
                <h4>规则 3</h4>
                <p>当检测到实体A为"梅西"时，可以确定其存在"效力于"的关联关系，该判断的置信度为100%。</p>
            </div>
        `;
    } else if (task === 'log') {
        inputQuery.value = '分析以下日志序列是否存在异常：\n[2023-02-01 08:01:23] INFO: 系统启动\n[2023-02-01 08:02:45] ERROR: 数据库连接失败\n[2023-02-01 08:03:12] INFO: 重试连接';
        
        // 加载示例规则
        const rulesList = document.getElementById('rules-list');
        rulesList.innerHTML = `
            <div class="rule-item">
                <h4>规则 1</h4>
                <p>当日志中出现ERROR级别消息后，应当有对应的恢复或重试操作（准确度：0.95）。</p>
            </div>
            <div class="rule-item">
                <h4>规则 2</h4>
                <p>系统启动后若出现连续3次以上的ERROR，判定为异常状态（准确度：0.98）。</p>
            </div>
        `;
    }
    
    // 清空输出区域
    document.getElementById('output-original').innerHTML = '';
    document.getElementById('output-enhanced').innerHTML = '';
}

// 运行演示
function runDemo() {
    const model = document.getElementById('model-select').value;
    const query = document.getElementById('input-query').value;
    
    // 显示加载状态
    document.getElementById('pipeline-viz').innerHTML = '<div class="loading">处理中...</div>';
    
    // 模拟API调用延迟
    setTimeout(() => {
        // 重绘处理流程，突出当前步骤
        highlightPipelineStep(0);
        
        // 模拟每个步骤的处理
        let step = 0;
        const interval = setInterval(() => {
            step++;
            if (step <= 8) {
                highlightPipelineStep(step);
            } else {
                clearInterval(interval);
                // 显示结果
                showResults();
            }
        }, 800);
    }, 1000);
}

// 高亮处理流程的当前步骤
function highlightPipelineStep(step) {
    const steps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];
    
    const pipelineDiagram = `
    graph LR
        A[用户查询] --> B[数据处理]
        B --> C[规则学习]
        C --> D[规则翻译]
        D --> E[规则整合]
        E --> F[规则增强生成]
        F --> G[规则验证]
        G --> H[文本调整]
        H --> I[最终输出]
        
        style C fill:#ff7f50,stroke:#333,stroke-width:2px
        style F fill:#ff7f50,stroke:#333,stroke-width:2px
        style G fill:#ff7f50,stroke:#333,stroke-width:2px
        style ${steps[step]} fill:#4CAF50,stroke:#333,stroke-width:4px
    `;
    
    document.getElementById('pipeline-viz').innerHTML = `<div class="mermaid">${pipelineDiagram}</div>`;
    mermaid.init(undefined, '.mermaid');
}

// 显示结果
function showResults() {
    const task = document.querySelector('.tab-btn.active').dataset.tab;
    
    if (task === 'relation') {
        // 原始LLM输出
        document.getElementById('output-original').innerHTML = `
            <p>在文本"梅西是巴塞罗那足球俱乐部的球员"中，我发现了以下实体关系：</p>
            <ul>
                <li>梅西 - 是 - 球员</li>
                <li>球员 - 属于 - 巴塞罗那足球俱乐部</li>
            </ul>
        `;
        
        // 规则增强输出
        document.getElementById('output-enhanced').innerHTML = `
            <p>在句子"梅西是巴塞罗那足球俱乐部的球员"中，可以提取以下实体关系：</p>
            <ol>
                <li><strong>实体</strong>
                    <ul>
                        <li><strong>梅西</strong>（类型：<code>人物</code>）</li>
                        <li><strong>巴塞罗那足球俱乐部</strong>（类型：<code>组织</code>）</li>
                    </ul>
                </li>
                <li><strong>关系</strong>
                    <ul>
                        <li><strong>关系类型</strong>：<code>效力于</code>（或<code>属于</code>）</li>
                        <li><strong>关系描述</strong>：梅西（人物）作为球员效力于巴塞罗那足球俱乐部（组织）。</li>
                    </ul>
                </li>
            </ol>
            <p><strong>结构化表示</strong>：</p>
            <pre>{
  "entities": [
    {"name": "梅西", "type": "人物"},
    {"name": "巴塞罗那足球俱乐部", "type": "组织"}
  ],
  "relations": [
    {
      "subject": "梅西",
      "predicate": "效力于",
      "object": "巴塞罗那足球俱乐部",
      "context": "球员身份"
    }
  ]
}</pre>
        `;
    } else if (task === 'log') {
        // 原始LLM输出
        document.getElementById('output-original').innerHTML = `
            <p>日志序列中存在异常，因为有ERROR级别的消息："数据库连接失败"。</p>
        `;
        
        // 规则增强输出
        document.getElementById('output-enhanced').innerHTML = `
            <p>分析结果：日志序列中存在<strong>轻微异常</strong>，但已有恢复措施。</p>
            <p><strong>异常详情</strong>：</p>
            <ul>
                <li>检测到ERROR级别消息："数据库连接失败"</li>
                <li>异常后有对应的恢复操作："重试连接"</li>
                <li>异常未连续出现3次以上，不构成严重异常状态</li>
            </ul>
            <p><strong>建议操作</strong>：监控后续日志，确认重试连接是否成功。若连接持续失败，应检查数据库服务状态。</p>
        `;
    }
}