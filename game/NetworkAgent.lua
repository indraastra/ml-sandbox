local socket = require "socket"

local NetworkAgent = Object:extend()
 
function NetworkAgent:new(x, y, opts)
    self.x, self.y = x, y
    local opts = opts or {} -- this handles the case where opts is nil
    for k, v in pairs(opts) do self[k] = v end

    udp = socket.udp()
    udp:settimeout(0)
    udp:setsockname(self.address, self.port)
end

function NetworkAgent:update(dt)
    repeat
        data, msg = udp:receive()
 
        if data then
            local x, y = data:match("^(%-?[%d.e]*) (%-?[%d.e]*)$")
            assert(x and y)
            self.x, self.y = tonumber(x), tonumber(y)
        end
    until not data
end

function NetworkAgent:draw()
end

function NetworkAgent:getPosition()
    return self.x, self.y
end

return NetworkAgent
