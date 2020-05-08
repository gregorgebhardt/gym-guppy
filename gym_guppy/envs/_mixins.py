from ..envs import GoalGuppyEnv, GuppyEnv
from ..guppies import BaseCouzinGuppy


class RenderCouzinZonesMixin(GuppyEnv):
    def _draw_on_table(self, screen):
        super(RenderCouzinZonesMixin, self)._draw_on_table(screen)

        for g in self.guppies:
            if isinstance(g, BaseCouzinGuppy):
                zor, zoo, zoa = g.couzin_zones

                width = .002
                screen.draw_circle(g.get_position(), zor + zoo + zoa, color=(0, 100, 0), filled=False, width=width)
                if zoo + zor > width:
                    screen.draw_circle(g.get_position(), zor + zoo, color=(50, 100, 100), filled=False, width=width)
                if zor > width:
                    screen.draw_circle(g.get_position(), zor, color=(100, 0, 0), filled=False, width=width)


class RenderGoalMixin(GoalGuppyEnv):
    def _draw_on_table(self, screen):
        super(RenderGoalMixin, self)._draw_on_table(screen)

        screen.draw_circle(self.desired_goal, self.change_goal_threshold, color=(100, 150, 220, 100), filled=True)
