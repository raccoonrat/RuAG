/**
 * 规则增强效果对比可视化
 * 展示原始LLM输出与规则增强输出的差异
 */
class ComparisonVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }
    
    /**
     * 可视化文本差异
     * @param {string} original 原始文本
     * @param {string} enhanced 增强文本
     */
    visualizeDifference(original, enhanced) {
        // 清空容器
        this.container.innerHTML = '';
        
        // 创建对比容器
        const comparisonDiv = document.createElement('div');
        comparisonDiv.className = 'comparison-container';
        
        // 创建原始文本容器
        const originalDiv = document.createElement('div');
        originalDiv.className = 'text-container original';
        originalDiv.innerHTML = `
            <h3>原始LLM输出</h3>
            <div class="text-content">${this.formatText(original)}</div>
        `;
        
        // 创建增强文本容器
        const enhancedDiv = document.createElement('div');
        enhancedDiv.className = 'text-container enhanced';
        enhancedDiv.innerHTML = `
            <h3>规则增强输出</h3>
            <div class="text-content">${this.formatText(enhanced)}</div>
        `;
        
        // 添加到对比容器
        comparisonDiv.appendChild(originalDiv);
        comparisonDiv.appendChild(enhancedDiv);
        
        // 添加差异高亮
        this.highlightDifferences(originalDiv.querySelector('.text-content'), enhancedDiv.querySelector('.text-content'));
        
        // 添加对比指标
        const metricsDiv = document.createElement('div');
        metricsDiv.className = 'comparison-metrics';
        metricsDiv.innerHTML = this.calculateMetrics(original, enhanced);
        
        // 添加到容器
        this.container.appendChild(comparisonDiv);
        this.container.appendChild(metricsDiv);
    }
    
    /**
     * 格式化文本
     * @param {string} text 文本
     * @returns {string} 格式化后的HTML
     */
    formatText(text) {
        // 处理JSON格式
        if (text.includes('{') && text.includes('}')) {
            try {
                // 尝试提取JSON部分
                const jsonMatch = text.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    const jsonPart = jsonMatch[0];
                    const parsedJson = JSON.parse(jsonPart);
                    const formattedJson = JSON.stringify(parsedJson, null, 2);
                    return text.replace(jsonPart, `<pre class="json-content">${this.escapeHtml(formattedJson)}</pre>`);
                }
            } catch (e) {
                // JSON解析失败，按普通文本处理
            }
        }
        
        // 处理普通文本
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }
    
    /**
     * 转义HTML特殊字符
     * @param {string} text 文本
     * @returns {string} 转义后的文本
     */
    escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
    
    /**
     * 高亮文本差异
     * @param {HTMLElement} originalEl 原始文本元素
     * @param {HTMLElement} enhancedEl 增强文本元素
     */
    highlightDifferences(originalEl, enhancedEl) {
        // 简单实现：标记增强文本中的关键词和结构
        const keywords = ['实体', '关系', '类型', '效力于', '属于', '人物', '组织', '结构化'];
        
        keywords.forEach(keyword => {
            const regex = new RegExp(`(${keyword})`, 'g');
            enhancedEl.innerHTML = enhancedEl.innerHTML.replace(
                regex, 
                '<span class="highlight">$1</span>'
            );
        });
    }
    
    /**
     * 计算对比指标
     * @param {string} original 原始文本
     * @param {string} enhanced 增强文本
     * @returns {string} 指标HTML
     */
    calculateMetrics(original, enhanced) {
        // 计算一些简单指标
        const originalLength = original.length;
        const enhancedLength = enhanced.length;
        const lengthDiff = ((enhancedLength - originalLength) / originalLength * 100).toFixed(1);
        
        // 结构化信息检测
        const hasStructuredInfo = enhanced.includes('结构化') || enhanced.includes('json') || enhanced.includes('{');
        
        // 实体关系检测
        const entityCount = (enhanced.match(/实体/g) || []).length;
        const relationCount = (enhanced.match(/关系/g) || []).length;
        
        return `
            <h3>增强效果指标</h3>
            <ul>
                <li>内容丰富度: <span class="metric ${lengthDiff > 0 ? 'positive' : 'negative'}">${lengthDiff > 0 ? '+' : ''}${lengthDiff}%</span></li>
                <li>结构化信息: <span class="metric ${hasStructuredInfo ? 'positive' : 'negative'}">${hasStructuredInfo ? '有' : '无'}</span></li>
                <li>实体识别: <span class="metric positive">${entityCount}个</span></li>
                <li>关系抽取: <span class="metric positive">${relationCount}个</span></li>
            </ul>
        `;
    }
}