function new_cursor = next_char(text, cursor)
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
        
        ch = text(current_cursor);
        if(ch >= 32 && ch <= 126)
            new_cursor = current_cursor;
            return;
        end        
    end
end

