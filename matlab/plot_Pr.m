clear

trial = 28;
K = 20;
N = 1;
B = 0;
E = 1;

Pr_set = [0.01, 0.1, 0.3, 0.5, 1];


filename=['training_result/cmp_Pr_trial_' num2str(trial) '_K_' num2str(K) '_N_' num2str(N)  '_B_' num2str(B) '_E_' num2str(E) '.mat'];

load(filename);

filename=['training_result/cmp_Pr_trial_' num2str(trial) '_K_' num2str(K) '_N_' num2str(N)  '_B_' num2str(B) '_E_' num2str(E) '_NMSE.mat'];

load(filename);

linesize=1.5;
MarkerSize=8;
LineWidth=1.5;

figure

hold on
plot(Pr_set, test_accuracy(1, 1 : end), 'k--', 'LineWidth', LineWidth, 'MarkerSize', MarkerSize);
plot(Pr_set, test_accuracy(2, 1 : end), 'r-o', 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerFaceColor', 'r');
plot(Pr_set, ones(length(Pr_set), 1) * mean(test_accuracy(3, 1: end)), '-^', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerFaceColor', [0.4940 0.1840 0.5560]);
plot(Pr_set, test_accuracy(5, 1 : end), '-p', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', LineWidth, 'MarkerSize', 2 + MarkerSize, 'MarkerFaceColor', [0.4660 0.6740 0.1880]);

set(get(gca, 'Children'), 'linewidth', 1.5)
set(gca, 'XTick', [0.01, 0.1, 0.3, 0.5, 1])
% xticklabels({'0.01', '0.1', '0.3', '0.5', '1'})
% set(gca, 'XLim', [Pr_set(1), Pr_set(end)])
set(gca, 'YTick', 0: 0.1: 0.9)
axis([0 Pr_set(end) 0 0.9])
axis([0.01 1 0.4 0.9])

grid on 
box on
hl = legend('Error-free channel', 'Proposed scheme', 'Conventional scheme', 'Existing scheme [26]');
set(hl,'Interpreter', 'latex', 'fontsize', 12, 'location', 'southeast')
xlabel('Maximum Relay Transmit Power $P_r$ (W)', 'Interpreter', 'latex', 'fontsize', 14);
ylabel('Test Accuracy','Interpreter', 'latex', 'fontsize', 14);

figure

hold on
plot(Pr_set, nmse(2, 1 : end), 'r-o', 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerFaceColor', 'r');
plot(Pr_set, ones(length(Pr_set), 1) * mean(nmse(3, 1: end)), '-^', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', LineWidth, 'MarkerSize', MarkerSize, 'MarkerFaceColor', [0.4940 0.1840 0.5560]);
plot(Pr_set, nmse(5, 1 : end), '-p', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', LineWidth, 'MarkerSize', 2 + MarkerSize, 'MarkerFaceColor', [0.4660 0.6740 0.1880]);

set(get(gca, 'Children'), 'linewidth', 1.5)
set(gca, 'XTick', [0.01, 0.1, 0.3, 0.5, 1])
% xticklabels({'0.01', '0.1', '0.3', '0.5', '1'})
set(gca, 'XLim', [Pr_set(1), Pr_set(end)])
% set(gca, 'YTick', -10: 5: 10)
% axis([0 Pr_set(end) 0 0.9])
axis([0.01 1 -10 10])

grid on 
box on
hl = legend('Proposed scheme', 'Conventional scheme', 'Existing scheme [26]');
set(hl,'Interpreter', 'latex', 'fontsize', 12, 'location', 'southeast')
xlabel('Maximum Relay Transmit Power $P_r$ (W)', 'Interpreter', 'latex', 'fontsize', 14);
ylabel('Average NMSE (dB)','Interpreter', 'latex', 'fontsize', 14);