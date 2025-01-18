% 双轴折线+柱状图绘制模板
% 公众号：阿昆的科研日常

clear
%% 数据准备
% 读取数据
A = load('1.txt');

% 自变量x
x = A(:,1);

% 左轴-柱状图，因变量bar1
bar1(:,1) = A(:,2);
bar1(:,2) = A(:,5);
bar1 = flipud(bar1); % 数据翻转，可删

% 右轴-折线图，因变量y1和y2
y1 = A(:,3);
y2 = A(:,6);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 14;
figureHeight = 8;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); 
hold on

%% 颜色定义
% colorplus函数获取方式：
% 公众号(阿昆的科研日常)后台回复：450
C1 = addcolorplus(3);
C2 = addcolorplus(4);

%% 定义左轴柱状图
yyaxis left
GO = bar(bar1,1,'EdgeColor','k');
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
hYLabel1 = ylabel('Number of points');
set(gca,  'YColor', [.1 .1 .1],...         % 坐标轴颜色
          'YTick', 0:5000:25000,...        % 刻度位置、间隔
          'Ylim' ,[0 25000],...            % 坐标轴范围
          'Yticklabel',{[0:5000:25000]})   % Y坐标轴刻度标签

%% 定义右轴折线图
yyaxis right
Ag = line(x,y1);
Ao = line(x,y2);
set(Ag, 'LineStyle', '-','LineWidth', 1,  'Color', C1)
set(Ao, 'LineStyle', '--','LineWidth', 1,  'Color', C2)
hYLabel2 = ylabel('Accuracy (%)');
set(gca, 'YColor', [.1 .1 .1],... 
         'YTick', 0:10:100, ...
         'Ylim' ,[50 100], ...
         'Yticklabel',{[0:10:100]})

     
%% X轴与其它细节调整
hXLabel1 = xlabel('cr (m)');
set(gca, 'Box', 'off', ...                                         % 边框
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],...                                  % 坐标轴颜色
         'XDir','reverse')                                         % X轴翻转（可删）
     
hLegend = legend([GO(1),GO(2),Ag,Ao], ...
    'Ground', 'Object', 'Pg','Po', ...
    'Location', 'southoutside','orientation','horizontal');

% 刻度标签字体和字号
set(gca, 'FontName', 'Helvetica', 'FontSize', 10)

figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r600','-dpng');