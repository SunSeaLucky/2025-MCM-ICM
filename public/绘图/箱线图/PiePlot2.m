% 带偏移扇区的饼图绘制模板
% 公众号：阿昆的科研日常

%% 数据准备
% 读取数据
X = [0.49 0.12 0.1 0.08 0.08 0.04 0.04 0.05];
explode = [0 0 1 0 1 0 0 0];
Label = {'Brazil','Colombia','Argentina','Peru','Venezuela','Chile','Ecuador','Others'};

%% 颜色定义
% addcolorplus函数获取方式：
% 公众号后台回复：配色强化
% 渐变色
map = addcolorplus(282);
idx = linspace(1,64,8);
idx = round(idx);
C = map(idx,:);
C = flipud(C);

%% 绘制带偏移扇区的饼图
pie(X,explode)
hTitle = title('South American Population by Country');

%% 细节优化
% 赋色
colormap(C)
% 标签
hLegend = legend(Label, 'Position', [0.85 0.15 0.1 0.3]);
hLegend.ItemTokenSize = [5 5];
legend('boxoff');
% 字体字号
th = findobj(gca, 'Type', 'text');
set(th, 'FontName', 'Arial', 'FontSize', 11)
set(hLegend, 'FontName',  'Arial', 'FontSize', 9)
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
print('test.png','-r300','-dpng');