function y = prox_op_l1(x, threshold)
    v = sign(x);
    y = v.*max(0, v.*(x - v*threshold));
%     for i = 1:len
%         if (x(i)>threshold) 
%             y(i) = x(i) - threshold;
%         elseif (x(i) < -threshold)
%             y(i) = x(i) + threshold;
%         else y(i) = 0;
%         end
%     end
end