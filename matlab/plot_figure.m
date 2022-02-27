clear

trial = 30;
K = 20;
N = 1;
B = 0;
E = 1;
lr = 0.05;

filename=['training_result/cmp_time_trial_' num2str(trial) '_K_' num2str(K) '_N_' num2str(N)  '_B_' num2str(B) '_E_' num2str(E) '.mat'];

load(filename);

index1 = 0 : length(test_accuracy1) - 1;
index2 = 0 : 2: length(test_accuracy1) - 1;

linesize=1.5;
MarkerSize=8;
LineWidth=1.5;

figure

hold on
plot(index1, test_accuracy1, 'k--', 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerIndices', 1: 10: length(index1));
plot(index2, test_accuracy2, 'r-o', 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerFaceColor', 'r', 'MarkerIndices', 1: 100: length(index2));
plot(index1, test_accuracy3, '-^', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerFaceColor', [0.4940 0.1840 0.5560], 'MarkerIndices', 1: 100: length(index1));
plot(index2, test_accuracy5, '-p', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', LineWidth, 'MarkerSize', 2 + MarkerSize, 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerIndices', 1: 100: length(index2));

set(get(gca, 'Children'), 'linewidth', 1.5)
set(gca, 'XTick', 0: 200: length(index1))
% set(gca, 'XLim', [K_set(1), K_set(end)])
set(gca, 'YTick', 0: 0.1: 0.9)
axis([index1(1) index1(end) 0 0.9])

grid on 
box on
hl = legend('Error-free channel', 'Proposed scheme', 'FL without relays [29]', 'Relay-assisted scheme in [22]');
set(hl,'Interpreter', 'latex', 'fontsize', 12, 'location', 'southeast')
xlabel('Number of Transmission Blocks', 'Interpreter', 'latex', 'fontsize', 14);
ylabel('Test Accuracy','Interpreter', 'latex', 'fontsize', 14);