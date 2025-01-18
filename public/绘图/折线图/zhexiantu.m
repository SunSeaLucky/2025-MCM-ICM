% 折线图绘制模板
% By：阿昆的科研日常
clear;
%% 数据准备
% 读取数据
load 折线图数据.mat; % 读入数据mae，mae中包括78*3的矩阵A

% 定义线上点的x坐标
x = 0.1:0.1:1;

% 分别定义不同线条上点的y坐标
samp1 = A(2:11,3);
samp2 = A(15:24,3);
samp3 = A(28:37,3);
samp4 = A(41:50,3);
samp5 = A(54:63,3);

%% 颜色定义
% addcolor函数获取方式：
% 公众号后台回复：morecolor
C1 = addcolorplus(1);
C2 = addcolorplus(98);
C3 = addcolorplus(135);
C4 = addcolorplus(214);
C5 = addcolorplus(215);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 12;
figureHeight = 8;

%% 线条绘制
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); 
hold on

% 分别绘制直线
Line1 = line(x,samp1);
Line2 = line(x,samp2);
Line3 = line(x,samp3);
Line4 = line(x,samp4);
Line5 = line(x,samp5);

%% 细节调整
% 定义线型、标记符号、线宽和颜色
% —————————————————————
% 线型 |   -   |    --   |    :   |   :.  |
% —————————————————————
% 说明 |  实线 |  双划线  |  虚线  | 点划线|
% —————————————————————

% ————————————————————————————————————————————————————————
% 标记符 |   +   |     o    |    *   |    .   |   x   |   s   |    d   |   ^    |    v   |     >   |    <   |
% ————————————————————————————————————————————————————————
%  说明  |  加号 |   空心圆  |  星号  | 实心圆 |  叉号 | 正方形 |  菱形  | 上三角 | 下三角  |  左三角 | 右三角 |  
% ————————————————————————————————————————————————————————
set(Line1, 'LineStyle', '--','Marker', 'd','LineWidth', 1,  'Color', C1)
set(Line2, 'LineStyle', '--','Marker', 'o','LineWidth', 1,  'Color', C2)
set(Line3, 'LineStyle', '--','Marker', '^','LineWidth', 1,  'Color', C3)
set(Line4, 'LineStyle', '--','Marker', 'v','LineWidth', 1,  'Color', C4)
set(Line5, 'LineStyle', '--','Marker', 'd','LineWidth', 1,  'Color', C5)

% 坐标区调整
set(gca, 'Box', 'off', ...                                % 边框
         'XGrid', 'off', 'YGrid', 'on', ...               % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...   % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...    % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...  % 坐标轴颜色
         'XTick', 0:0.1:1.2,  'YTick', 0:0.05:0.5,...     % 刻度位置、间隔
         'Xlim' ,[0 1.1],'Ylim' ,[0.1 0.35], ...          % 坐标轴范围
         'Xticklabel',{[0:0.1:1.2]},...                   % X坐标轴刻度标签
         'Yticklabel',{[0:0.05:0.5]})                     % Y坐标轴刻度标签

% X、Y轴标签及Legend
hXLabel = xlabel('Point spacing (m)');
hYLabel = ylabel('MAE (m)');
hLegend = legend([Line1,Line2,Line3,Line4,Line5], ...
                 'samp1', 'samp2','samp3','samp4','samp5', ...
                 'Location', 'northwest');

% 刻度标签字体和字号
set(gca, 'FontName', 'Times', 'FontSize', 9)
% X、Y轴标签及Legend的字体字号 
set([hXLabel, hYLabel,hLegend], 'FontName',  'Helvetica')
set([hXLabel, hYLabel,hLegend], 'FontSize', 10)

% 背景颜色
set(gca,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'eg';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');

