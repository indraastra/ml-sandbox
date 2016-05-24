-- in Ship.lua
local Ship = Object:extend()

function Ship:new(x, y, opts)
    self.x, self.y = x, y
    self.lin_speed = 3
    self.rot_speed = .05
    self.angle = 0
    self.dead = false
    self.controller = love.mouse

    local opts = opts or {} -- this handles the case where opts is nil
    for k, v in pairs(opts) do self[k] = v end
end

function Ship:update(dt)
    local target_x, target_y = self.controller:getPosition()
    local target_angle = math.atan2(target_y - self.y, target_x - self.x)

    local da_pos = (target_angle - self.angle) % (math.pi * 2)
    local da_neg = (self.angle - target_angle) % (math.pi * 2)

    local da = 0
    if da_pos < da_neg then
        da = da_pos
    else
        da = -da_neg
    end

    self.angle = self.angle + math.max(math.min(self.rot_speed, da), -self.rot_speed)
    self.angle = self.angle % (math.pi * 2)
    self.x = self.x + math.cos(self.angle) * self.lin_speed
    self.y = self.y + math.sin(self.angle) * self.lin_speed
end

function Ship:draw()
    love.graphics.push()
    love.graphics.translate(self.x, self.y)
    love.graphics.rotate(self.angle - math.pi/2)
    love.graphics.polygon('fill', -15, 0, 15, 0, 0, 50)
    love.graphics.pop()
end

return Ship

