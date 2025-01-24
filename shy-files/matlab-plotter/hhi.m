% ��ȡ CSV �ļ�  
data = readtable('data_hhi.csv'); % �����ļ���Ϊ data.csv  

% ��ȡ��ݺ� HHIndex ����  
years = data.Year;  
hhIndex = data.HHIndex;  

% ����ͼ��  
figure;  
plot(years, hhIndex, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);  
grid on;  

% ����ͼ�α�������ǩ  
title('HHIndex Over the Years');  
xlabel('Year');  
ylabel('HHIndex');  

% ���� x ��̶�  
xticks(years);  
xtickangle(45); % ��ת x ��̶��Ա��������ʾ  

% ��ʾͼ��  
legend('HHIndex', 'Location', 'best');