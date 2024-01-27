import { z } from "zod"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const locationRouter = createTRPCRouter({
  saveLocation: protectedProcedure
    .input(z.object({}))
    .mutation(async ({ ctx, input }) => {
      return true
    }),
})
