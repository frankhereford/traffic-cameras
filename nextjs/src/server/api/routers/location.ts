import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const locationRouter = createTRPCRouter({
  saveLocation: protectedProcedure
    .input(
      z.object({
        correlatedLocation: z.object({
          x: z.number(),
          y: z.number(),
          latitude: z.number(),
          longitude: z.number(),
        }),
        camera: z.number(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      console.log("saveLocation", input)
      const camera = await ctx.db.camera.findFirstOrThrow({
        where: { coaId: input.camera },
      })
      await ctx.db.location.create({
        data: {
          x: input.correlatedLocation.x,
          y: input.correlatedLocation.y,
          latitude: input.correlatedLocation.latitude,
          longitude: input.correlatedLocation.longitude,
          camera: { connect: { id: camera.id } },
          user: {
            connect: { id: ctx.session.user.id },
          },
        },
      })
      return true
    }),
})
