/**
 * RuAG演示流程控制
 * 协调各个可视化组件，展示完整的RuAG处理流程
 */
class DemoFlow {
    constructor() {
        // 初始化各个可视化组件
        this.ruleVisualizer = new RuleVisualizer('rules-viz');
        this.mctsVisualizer = new MCTSVisualizer('mcts-viz');
        this.comparisonVisualizer = new ComparisonVisualizer('comparison-viz');
        
        // 当前演示状态
        this.currentStep = 0;
        this.demoRunning = false;
        
        // 绑定事件
        this.bindEvents();
    }
    
    /**
     * 绑定事件处理
     */
    bindEvents() {
        // 运行按钮
        document.getElementById('run-btn').addEventListener('click', () => {
            this.startDemo();
        });
        
        // 步骤按钮
        document.getElementById('next-step-btn').addEventListener('click', () => {
            this.nextStep();
        });
        
        document.getElementById('prev-step-btn').addEventListener('click', () => {
            this.prevStep();
        });
        
        // 任务切换
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.resetDemo();
            });
        });
    }
    
    /**
     * 开始演示
     */
    startDemo() {
        if (this.demoRunning) return;
        
        this.demoRunning = true;
        this.currentStep = 0;
        
        // 获取输入
        const task = document.querySelector('.tab-btn.active').dataset.tab;
        const model = document.getElementById('model-select').value;
        const query = document.getElementById('input-query').value;
        
        // 显示加载状态
        document.getElementById('status-message').textContent = '正在处理...';
        document.getElementById('step-indicator').textContent = '步骤 1/8: 初始化';
        
        // 发送API请求
        fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ task, model, query })
        })
        .then(response => response.json())
        .then(data => {
            // 保存结果数据
            this.demoData = data;
            
            // 更新API_KEY显示
            document.getElementById('api-key-display').value = data.modelApiKey;
            
            // 更新状态
            document.getElementById('status-message').textContent = '演示就绪';
            
            // 显示第一步
            this.showStep(0);
        })
        .catch(error => {
            console.error('演示运行失败:', error);
            document.getElementById('status-message').textContent = '演示失败: ' + error.message;
            this.demoRunning = false;
        });
    }
    
    /**
     * 重置演示
     */
    resetDemo() {
        this.demoRunning = false;
        this.currentStep = 0;
        
        // 重置UI
        document.getElementById('status-message').textContent = '准备就绪';
        document.getElementById('step-indicator').textContent = '';
        
        // 清空可视化区域
        document.getElementById('pipeline-viz').innerHTML = '';
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('comparison-viz').innerHTML = '';
        
        // 重置步骤按钮
        document.getElementById('next-step-btn').disabled = false;
        document.getElementById('prev-step-btn').disabled = true;
    }
    
    /**
     * 下一步
     */
    nextStep() {
        if (!this.demoRunning || this.currentStep >= 7) return;
        
        this.currentStep++;
        this.showStep(this.currentStep);
    }
    
    /**
     * 上一步
     */
    prevStep() {
        if (!this.demoRunning || this.currentStep <= 0) return;
        
        this.currentStep--;
        this.showStep(this.currentStep);
    }
    
    /**
     * 显示指定步骤
     * @param {number} step 步骤索引
     */
    showStep(step) {
        // 更新步骤指示器
        const stepNames = [
            '初始化', '数据处理', '规则学习', '规则翻译', 
            '规则整合', '规则增强生成', '规则验证', '最终输出'
        ];
        
        document.getElementById('step-indicator').textContent = `步骤 ${step + 1}/8: ${stepNames[step]}`;
        
        // 更新步骤按钮状态
        document.getElementById('prev-step-btn').disabled = (step === 0);
        document.getElementById('next-step-btn').disabled = (step === 7);
        
        // 高亮当前步骤
        this.highlightPipelineStep(step);
        
        // 根据步骤显示不同内容
        switch (step) {
            case 0: // 初始化
                this.showInitialization();
                break;
            case 1: // 数据处理
                this.showDataProcessing();
                break;
            case 2: // 规则学习
                this.showRuleLearning();
                break;
            case 3: // 规则翻译
                this.showRuleTranslation();
                break;
            case 4: // 规则整合
                this.showRuleIntegration();
                break;
            case 5: // 规则增强生成
                this.showRuleAugmentedGeneration();
                break;
            case 6: // 规则验证
                this.showRuleValidation();
                break;
            case 7: // 最终输出
                this.showFinalOutput();
                break;
        }
    }
    
    /**
     * 高亮处理流程的当前步骤
     * @param {number} step 步骤索引
     */
    highlightPipelineStep(step) {
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
    
    /**
     * 显示初始化步骤
     */
    showInitialization() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>1. 初始化LLM提供者</h3>
            <p>当前使用模型: <strong>${this.demoData.model}</strong></p>
            <p>任务类型: <strong>${this.demoData.task === 'relation' ? '关系抽取' : '日志异常检测'}</strong></p>
            <p>查询内容:</p>
            <pre>${this.demoData.query}</pre>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 清空其他可视化区域
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('comparison-viz').innerHTML = '';
    }
    
    /**
     * 显示数据处理步骤
     */
    showDataProcessing() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>2. 数据预处理</h3>
            <p>在这一步骤中，系统对输入数据进行预处理，为规则学习做准备。</p>
            <p>预处理包括:</p>
            <ul>
                <li>文本标准化</li>
                <li>实体识别</li>
                <li>关系标注</li>
                <li>特征提取</li>
            </ul>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 在MCTS可视化区域显示数据示例
        const dataDiv = document.createElement('div');
        dataDiv.className = 'data-examples';
        
        if (this.demoData.task === 'relation') {
            dataDiv.innerHTML = `
                <h3>数据示例</h3>
                <div class="data-item">
                    <p><strong>文本:</strong> "哈兰德是曼城足球俱乐部的前锋。"</p>
                    <p><strong>实体:</strong> ["哈兰德", "曼城足球俱乐部"]</p>
                    <p><strong>关系:</strong> "效力于"</p>
                </div>
                <div class="data-item">
                    <p><strong>文本:</strong> "姆巴佩是巴黎圣日耳曼足球俱乐部的球员。"</p>
                    <p><strong>实体:</strong> ["姆巴佩", "巴黎圣日耳曼足球俱乐部"]</p>
                    <p><strong>关系:</strong> "效力于"</p>
                </div>
            `;
        } else {
            dataDiv.innerHTML = `
                <h3>数据示例</h3>
                <div class="data-item">
                    <p><strong>日志序列:</strong> ["系统启动", "数据库连接失败", "重试连接", "连接成功"]</p>
                    <p><strong>异常标记:</strong> true</p>
                </div>
                <div class="data-item">
                    <p><strong>日志序列:</strong> ["系统启动", "用户登录", "查询数据", "用户登出"]</p>
                    <p><strong>异常标记:</strong> false</p>
                </div>
            `;
        }
        
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('mcts-viz').appendChild(dataDiv);
        
        // 清空比较区域
        document.getElementById('comparison-viz').innerHTML = '';
    }
    
    /**
     * 显示规则学习步骤
     */
    showRuleLearning() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>3. 规则学习</h3>
            <p>使用蒙特卡洛树搜索(MCTS)算法从数据中自动提取高质量逻辑规则。</p>
            <p>学习到的规则数量: <strong>${this.demoData.rules.length}</strong></p>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 初始化MCTS可视化
        document.getElementById('mcts-viz').innerHTML = '<div class="mcts-container"></div>';
        this.mctsVisualizer = new MCTSVisualizer('mcts-viz');
        this.mctsVisualizer.initTree();
        
        // 模拟MCTS搜索过程
        setTimeout(() => {
            this.mctsVisualizer.simulateSearch(5);
        }, 500);
        
        // 在比较区域显示规则学习过程
        const processDiv = document.createElement('div');
        processDiv.className = 'learning-process';
        processDiv.innerHTML = `
            <h3>规则学习过程</h3>
            <div class="process-steps">
                <div class="process-step">
                    <div class="step-number">1</div>
                    <div class="step-desc">初始化搜索树</div>
                </div>
                <div class="process-step">
                    <div class="step-number">2</div>
                    <div class="step-desc">选择阶段：根据UCB值选择节点</div>
                </div>
                <div class="process-step">
                    <div class="step-number">3</div>
                    <div class="step-desc">扩展阶段：生成候选规则</div>
                </div>
                <div class="process-step">
                    <div class="step-number">4</div>
                    <div class="step-desc">模拟阶段：评估规则质量</div>
                </div>
                <div class="process-step">
                    <div class="step-number">5</div>
                    <div class="step-desc">回溯阶段：更新节点统计</div>
                </div>
                <div class="process-step">
                    <div class="step-number">6</div>
                    <div class="step-desc">重复步骤2-5直到收敛</div>
                </div>
                <div class="process-step">
                    <div class="step-number">7</div>
                    <div class="step-desc">选择最优规则</div>
                </div>
            </div>
        `;
        
        document.getElementById('comparison-viz').innerHTML = '';
        document.getElementById('comparison-viz').appendChild(processDiv);
    }
    
    /**
     * 显示规则翻译步骤
     */
    showRuleTranslation() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>4. 规则翻译</h3>
            <p>将机器可读的规则翻译成自然语言表述，提高可解释性。</p>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 显示规则翻译前后对比
        const rulesDiv = document.createElement('div');
        rulesDiv.className = 'rules-translation';
        
        let rulesHtml = '<h3>规则翻译示例</h3>';
        
        this.demoData.rules.slice(0, 3).forEach((rule, index) => {
            rulesHtml += `
                <div class="rule-translation">
                    <div class="rule-original">
                        <h4>原始规则 ${index + 1}</h4>
                        <pre>${JSON.stringify({
                            condition: rule.condition || `entity_a_${rule.name}`,
                            prediction: rule.prediction || 'relation_效力于',
                            accuracy: rule.accuracy || 1.0
                        }, null, 2)}</pre>
                    </div>
                    <div class="rule-arrow">→</div>
                    <div class="rule-translated">
                        <h4>翻译后规则 ${index + 1}</h4>
                        <p>${rule.translated || rule.name}</p>
                    </div>
                </div>
            `;
        });
        
        rulesDiv.innerHTML = rulesHtml;
        
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('mcts-viz').appendChild(rulesDiv);
        
        // 清空比较区域
        document.getElementById('comparison-viz').innerHTML = '';
    }
    
    /**
     * 显示规则整合步骤
     */
    showRuleIntegration() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>5. 规则整合</h3>
            <p>将学习到的规则整合到生成过程中，引导模型生成符合规则的文本。</p>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 显示规则整合过程
        const integrationDiv = document.createElement('div');
        integrationDiv.className = 'rule-integration';
        integrationDiv.innerHTML = `
            <h3>规则整合过程</h3>
            <div class="integration-flow">
                <div class="integration-item">
                    <h4>1. 规则检索</h4>
                    <p>根据查询上下文检索相关规则</p>
                </div>
                <div class="integration-arrow">↓</div>
                <div class="integration-item">
                    <h4>2. 规则优先级排序</h4>
                    <p>根据规则精确度和相关性排序</p>
                </div>
                <div class="integration-arrow">↓</div>
                <div class="integration-item">
                    <h4>3. 提示增强</h4>
                    <p>将规则整合到提示中</p>
                </div>
            </div>
        `;
        
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('mcts-viz').appendChild(integrationDiv);
        
        // 显示整合后的提示
        const promptDiv = document.createElement('div');
        promptDiv.className = 'enhanced-prompt';
        promptDiv.innerHTML = `
            <h3>增强后的提示</h3>
            <div class="prompt-content">
                <p><strong>原始查询:</strong></p>
                <pre>${this.demoData.query}</pre>
                <p><strong>整合规则后:</strong></p>
                <pre>请分析以下文本中的实体关系，并遵循这些规则:
1. ${this.demoData.rules[0].translated || '当实体A为哈兰德时，存在效力于关系（准确度：1.00）。'}
2. ${this.demoData.rules[1].translated || '当实体为姆巴佩时，可以确定其效力于关系，精确度为1.00。'}
3. ${this.demoData.rules[2].translated || '当检测到实体A为"梅西"时，可以确定其存在"效力于"的关联关系，该判断的置信度为100%。'}

${this.demoData.query}</pre>
            </div>
        `;
        
        document.getElementById('comparison-viz').innerHTML = '';
        document.getElementById('comparison-viz').appendChild(promptDiv);
    }
    
    /**
     * 显示规则增强生成步骤
     */
    showRuleAugmentedGeneration() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>6. 规则增强生成</h3>
            <p>使用整合了规则的提示生成文本，引导模型生成符合规则的输出。</p>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 显示生成过程
        const generationDiv = document.createElement('div');
        generationDiv.className = 'generation-process';
        generationDiv.innerHTML = `
            <h3>生成过程</h3>
            <div class="process-flow">
                <div class="process-item">
                    <h4>增强提示</h4>
                    <p>包含规则的提示</p>
                </div>
                <div class="process-arrow">↓</div>
                <div class="process-item highlight">
                    <h4>LLM生成</h4>
                    <p>模型: ${this.demoData.model}</p>
                </div>
                <div class="process-arrow">↓</div>
                <div class="process-item">
                    <h4>生成结果</h4>
                    <p>规则增强的输出</p>
                </div>
            </div>
        `;
        
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('mcts-viz').appendChild(generationDiv);
        
        // 显示生成结果对比
        this.comparisonVisualizer = new ComparisonVisualizer('comparison-viz');
        this.comparisonVisualizer.visualizeDifference(
            this.demoData.original_response,
            this.demoData.enhanced_response
        );
    }
    
    /**
     * 显示规则验证步骤
     */
    showRuleValidation() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>7. 规则验证</h3>
            <p>验证生成的文本是否符合规则，确保输出质量。</p>
            <p>验证结果: <strong>${this.demoData.is_valid ? '通过' : '不通过'}</strong></p>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 显示验证过程
        const validationDiv = document.createElement('div');
        validationDiv.className = 'validation-process';
        validationDiv.innerHTML = `
            <h3>验证过程</h3>
            <div class="validation-steps">
                <div class="validation-step">
                    <h4>1. 规则解析</h4>
                    <p>解析规则条件和预测</p>
                </div>
                <div class="validation-arrow">↓</div>
                <div class="validation-step">
                    <h4>2. 文本分析</h4>
                    <p>分析生成文本的内容</p>
                </div>
                <div class="validation-arrow">↓</div>
                <div class="validation-step highlight">
                    <h4>3. 规则匹配</h4>
                    <p>检查文本是否符合规则</p>
                </div>
                <div class="validation-arrow">↓</div>
                <div class="validation-step">
                    <h4>4. 验证结果</h4>
                    <p>${this.demoData.is_valid ? '所有规则验证通过' : '存在规则违反'}</p>
                </div>
            </div>
        `;
        
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('mcts-viz').appendChild(validationDiv);
        
        // 显示验证结果
        const resultDiv = document.createElement('div');
        resultDiv.className = 'validation-result';
        
        let resultHtml = `
            <h3>验证结果详情</h3>
            <div class="result-summary ${this.demoData.is_valid ? 'valid' : 'invalid'}">
                <div class="result-icon">${this.demoData.is_valid ? '✓' : '✗'}</div>
                <div class="result-text">
                    <h4>${this.demoData.is_valid ? '验证通过' : '验证失败'}</h4>
                    <p>${this.demoData.is_valid ? '生成的文本符合所有规则约束' : '生成的文本违反了一些规则'}</p>
                </div>
            </div>
        `;
        
        if (!this.demoData.is_valid && this.demoData.violations && this.demoData.violations.length > 0) {
            resultHtml += '<div class="violations-list"><h4>规则违反详情:</h4><ul>';
            this.demoData.violations.forEach(violation => {
                resultHtml += `<li>${violation}</li>`;
            });
            resultHtml += '</ul></div>';
        }
        
        resultDiv.innerHTML = resultHtml;
        
        document.getElementById('comparison-viz').innerHTML = '';
        document.getElementById('comparison-viz').appendChild(resultDiv);
    }
    
    /**
     * 显示最终输出步骤
     */
    showFinalOutput() {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'step-info';
        infoDiv.innerHTML = `
            <h3>8. 最终输出</h3>
            <p>生成的文本已通过规则验证，可以作为最终输出返回给用户。</p>
        `;
        
        document.getElementById('rules-viz').innerHTML = '';
        document.getElementById('rules-viz').appendChild(infoDiv);
        
        // 显示最终输出
        const outputDiv = document.createElement('div');
        outputDiv.className = 'final-output';
        outputDiv.innerHTML = `
            <h3>最终输出</h3>
            <div class="output-content">
                <pre>${this.demoData.enhanced_response}</pre>
            </div>
        `;
        
        document.getElementById('mcts-viz').innerHTML = '';
        document.getElementById('mcts-viz').appendChild(outputDiv);
        
        // 显示总结
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'demo-summary';
        summaryDiv.innerHTML = `
            <h3>演示总结</h3>
            <div class="summary-content">
                <p>RuAG框架通过自动学习规则并将其整合到生成过程中，成功提升了大型语言模型的输出质量。</p>
                <div class="key-benefits">
                    <h4>主要优势:</h4>
                    <ul>
                        <li>自动规则学习，无需人工定义</li>
                        <li>规则自然语言翻译，提高可解释性</li>
                        <li>规则增强生成，引导模型生成符合规则的文本</li>
                        <li>规则验证与调整，确保输出质量</li>
                        <li>多模型支持，提高框架通用性</li>
                    </ul>
                </div>
            </div>
        `;
        
        document.getElementById('comparison-viz').innerHTML = '';
        document.getElementById('comparison-viz').appendChild(summaryDiv);
    }
}

// 初始化演示流程
document.addEventListener('DOMContentLoaded', function() {
    window.demoFlow = new DemoFlow();
});