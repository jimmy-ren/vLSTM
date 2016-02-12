function new_cursor = next_word(text, cursor)
    if(cursor < 0)
        new_cursor = -1;
        return;
    end

    current_cursor = cursor;
    
    while(1)
        current_cursor = current_cursor + 1;
        if(current_cursor > length(text))
            new_cursor = -1;
            return;
        end
        
        % if a capital char
        ch = text(current_cursor);
        if(ch >= 65 && ch <= 90)
            new_cursor = current_cursor;
            return;
        end
        % if a space
        if(ch == 32)
            % the next valid char is the one
            new_cursor = next_letter(text, current_cursor);
            return;
        end        
    end
end

