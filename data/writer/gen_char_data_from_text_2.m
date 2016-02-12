addpath data/

% load all the text
% fid = fopen('data/graham/text.txt');
% tline = fgets(fid);
% all_text = tline;
% lc = 1;
% while ischar(tline)    
%     tline = fgets(fid);
%     all_text = strcat(all_text, tline);
%     lc = lc + 1;
%     fprintf('%d\n', lc);
% end
% fclose(fid);
load('data/writer/all_text.mat');

% use ascii code 32 to 126, 95 chars in all
seq_len = 50;
seq_num = 10000;
sample_seq = zeros(95, seq_len, seq_num);
label_seq = zeros(95, seq_len, seq_num);
fnum = 1;
cursor = 1;
while(1)    
    for m = 1:seq_num
        if(mod(m, 100) == 0)
            fprintf('processed %d sequences\n', m);
        end
        for n = 1:seq_len            
            ch = all_text(cursor);
            sample_seq(ch-31, n, m) = 1;
            % find the next valid char
            cursor = next_char(all_text, cursor);
            if(cursor < 0)
                break;
            end
            next_ch = all_text(cursor);
            label_seq(next_ch-31, n, m) = 1;
        end
        % 1 step back
        cursor = cursor - 1;
        % find the next starting word
        cursor = next_word(all_text, cursor);
        if(cursor < 0)
            break;
        end
    end
    if(cursor < 0)
        break;
    end
    
    % save
    fprintf('saving %d file.\n', fnum);
    save(strcat('data/writer/graham/seq_', num2str(fnum), '.mat'), '-v7.3', 'sample_seq', 'label_seq');
    fnum = fnum + 1;
    sample_seq = zeros(95, seq_len, seq_num);
    label_seq = zeros(95, seq_len, seq_num);
end






