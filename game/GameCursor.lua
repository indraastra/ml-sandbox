-- in GameCursor.lua
local GameCursor = Object:extend()

function GameCursor:new(x, y, opts)
    self.x, self.y = x, y
    self.dead = false
    self.controller = love.mouse

    local opts = opts or {} -- this handles the case where opts is nil
    for k, v in pairs(opts) do self[k] = v end

    timer.every(.01, function()
        createGameObject('Trail', self.x, self.y, {r = 5})
    end)

end

function GameCursor:update(dt)
    local x, y = self.controller:getPosition()
    self.x, self.y = x, y
end

function GameCursor:draw()
    love.graphics.circle('fill', self.x, self.y, 5)
end

return GameCursor
