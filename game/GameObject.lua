-- in Ship.lua
local Ship = Object:extend()

function Ship:new(x, y, opts)
    self.x, self.y = x, y
    local opts = opts or {} -- this handles the case where opts is nil
    for k, v in pairs(opts) do self[k] = v end
    self.dead = false

    timer.every(.01, function()
        createShip('Trail', self.x, self.y, {r = 10})
    end)

end

function Ship:update(dt)
    local x, y = love.mouse.getPosition()
    self.x, self.y = x, y
end

function Ship:draw()
    love.graphics.circle('fill', self.x, self.y, 10)
end

return Ship

