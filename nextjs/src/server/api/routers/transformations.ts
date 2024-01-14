import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const transformation = createTRPCRouter({
  submitWarpRequest: publicProcedure
    .input(
      z.object({
        points: z.array(
          z.object({
            cctvPoint: z.object({
              x: z.number(),
              y: z.number(),
            }),
            mapPoint: z.object({
              lat: z.number(),
              lng: z.number(),
            }),
          }),
        ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      console.log("input", JSON.stringify(input, null, 2))
      await new Promise((resolve) => setTimeout(resolve, 1000))
      console.log("done")
      return input.points.length
    }),

  getSecretMessage: protectedProcedure.query(() => {
    return "you can now see this secret message!"
  }),
})
