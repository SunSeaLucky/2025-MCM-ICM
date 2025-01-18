% 热图灵活版绘制模板
% 公众号：阿昆的科研日常

%% 数据准备
% 读取数据
rho=[1.00    0.16    0.29    0.05    0.34    0.41    0.29    0.22    0.25    0.56
     0.16    1.00    0.44    0.29    0.13    0.12    0.19    0.01    0.26    0.07
     0.29    0.44    1.00    0.08    0.32    0.35    0.36    0.20    0.02    0.27
     0.05    0.29    0.08    1.00    0.21    0.20    0.26    0.24    0.20    0.06
     0.34    0.13    0.32    0.21    1.00    0.86    0.45    0.61    0.06    0.43
     0.41    0.12    0.35    0.20    0.86    1.00    0.54    0.65    0.17    0.54
     0.29    0.19    0.36    0.26    0.45    0.54    1.00    0.37    0.14    0.26
     0.22    0.01    0.20    0.24    0.61    0.65    0.37    1.00    0.03    0.30
     0.25    0.26    0.02    0.20    0.06    0.17    0.14    0.03    1.00    0.52
     0.56    0.07    0.27    0.06    0.43    0.54    0.26    0.30    0.52    1.00];

%% 颜色定义
% addcolorplus函数获取方式：
% 公众号后台回复：配色强化
map = addcolorplus(303);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 12;
figureHeight = 11;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 热图灵活版绘制
b = bar3(rho,1);
hTitle = title('Correlation Coefficient');
hXLabel = xlabel('XLabel');
hYLabel = ylabel('YLabel');

%% 细节优化
% 赋色
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
colormap(map)
% 俯视
view(0,90)
% 添加颜色条
colorbar
% 消除坐标区缝隙
axis tight
% 坐标区调整
string_name={'S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'};
set(gca, 'Box', 'off', ...                                                          % 边框
         'LineWidth', 1, 'GridLineStyle', '-',...                                   % 坐标轴线宽
         'XGrid', 'off', 'YGrid', 'off','ZGrid', 'off', ...                         % 网格
         'TickDir', 'out', 'TickLength', [0 0], ...                                 % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off',  'ZMinorTick', 'off',...         % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1], 'ZColor', [.1 .1 .1],...      % 坐标轴颜色
         'Xticklabel',string_name,...                                               % X坐标轴刻度标签
         'Yticklabel',string_name)                                                  % Y坐标轴刻度标签
% 字体和字号
set(gca, 'FontName', 'Helvetica')
set([hXLabel, hYLabel], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 10)
set([hXLabel, hYLabel], 'FontSize', 12)
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');